"""Periodic KV-cache compression for reasoning LLMs: RewindKV and RPC.

Both methods compress the KV cache of *generated* tokens once every ``P``
decoding steps (prompt tokens are never evicted). After the n-th compression
the generated part of the cache keeps ``n * P / c`` tokens chosen by attention
scores computed from a set of *selector* queries:

* ``rpc``      : the selectors are the last ``R`` queries (Reasoning Path Compression).
* ``rewindkv`` : ``K`` selectors are chosen from all queries generated since the
                 last compression by block-wise selection that trades off
                 recency against within-block diversity (weight ``alpha``).
                 The tokens at the selector positions are always kept.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = ("rpc", "rewindkv")

# Default hyper-parameters of each method (overridable in set_compression_config).
METHOD_DEFAULTS = {
    "rpc": dict(R=32, scoring="attention_sum", aggregation="all", pooling="avgpool"),
    "rewindkv": dict(K=128, block_size=32, alpha=0.5, half_life=4,
                     scoring="attention_max", aggregation="group"),
}


def set_compression_config(
    model,
    method="rewindkv",
    P=1024,
    c=4,
    R=None,
    K=None,
    block_size=None,
    alpha=None,
    half_life=None,
    scoring=None,
    aggregation=None,
    pooling=None,
    kernel_size=7,
    ):
    """Configure KV-cache compression on every attention layer of ``model``.

    Args:
        method: ``"rewindkv"`` or ``"rpc"``.
        P: compression period (number of generated tokens between compressions).
        c: compression ratio; each compression round keeps ``P / c`` more tokens.
        R: (rpc) number of most recent queries used as selectors.
        K: (rewindkv) number of selector queries.
        block_size: (rewindkv) number of queries per block.
        alpha: (rewindkv) recency weight; ``1 - alpha`` weights diversity.
        half_life: (rewindkv) recency half-life, in blocks.
        scoring: ``"attention_sum"`` (sum over selector queries, mean over heads,
            followed by local pooling) or ``"attention_max"`` (max over selector
            queries and heads, no pooling).
        aggregation: ``"all"`` (one score shared by all KV heads) or ``"group"``
            (one score per KV head, aggregated over its query-head group).
        pooling: ``"avgpool"`` or ``"maxpool"`` (``attention_sum`` only).
        kernel_size: local pooling kernel size (``attention_sum`` only).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    cfg = dict(METHOD_DEFAULTS[method])
    overrides = dict(R=R, K=K, block_size=block_size, alpha=alpha, half_life=half_life,
                     scoring=scoring, aggregation=aggregation, pooling=pooling)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg.setdefault("pooling", "avgpool")

    if cfg["scoring"] not in ("attention_sum", "attention_max"):
        raise ValueError(f"Unsupported scoring policy: {cfg['scoring']}")
    if cfg["aggregation"] not in ("all", "group"):
        raise ValueError(f"Unsupported aggregation policy: {cfg['aggregation']}")
    if cfg["pooling"] not in ("avgpool", "maxpool"):
        raise ValueError(f"Unsupported pooling policy: {cfg['pooling']}")
    if P % c != 0:
        raise ValueError("P must be divisible by c")

    for layer in model.model.layers:
        cluster = layer.self_attn.kv_cluster
        cluster.method = method
        cluster.P = P
        cluster.c = c
        cluster.T = P // c
        cluster.scoring = cfg["scoring"]
        cluster.aggregation = cfg["aggregation"]
        cluster.pooling = cfg["pooling"]
        cluster.kernel_size = kernel_size
        if method == "rpc":
            # R recent queries are the selectors and the R most recent tokens are kept.
            cluster.R = cfg["R"]
        else:
            # The K selector tokens play the role of R in the compression schedule.
            cluster.R = cfg["K"]
            cluster.K = cfg["K"]
            cluster.block_size = cfg["block_size"]
            cluster.alpha = cfg["alpha"]
            cluster.half_life = cfg["half_life"]

    if method == "rpc":
        print(f"[KV compression] method=rpc P={P} c={c} R={cfg['R']} "
              f"scoring={cfg['scoring']} aggregation={cfg['aggregation']} pooling={cfg['pooling']}", flush=True)
    else:
        print(f"[KV compression] method=rewindkv P={P} c={c} K={cfg['K']} block_size={cfg['block_size']} "
              f"alpha={cfg['alpha']} half_life={cfg['half_life']} "
              f"scoring={cfg['scoring']} aggregation={cfg['aggregation']}", flush=True)


# Copied from transformers.models.llama.modeling_llama.repeat_kv for gqa_support
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class KVCluster():
    """Per-layer compression state and selection logic."""

    def __init__(self, layer_idx=None, num_key_value_groups=1):
        self.layer_idx = layer_idx
        self.num_key_value_groups = num_key_value_groups

        # Defaults; overwritten by set_compression_config.
        self.method = "rewindkv"
        self.P = 1024
        self.c = 4
        self.T = self.P // self.c
        self.R = 128
        self.K = 128
        self.block_size = 32
        self.alpha = 0.5
        self.half_life = 4
        self.scoring = "attention_max"
        self.aggregation = "group"
        self.pooling = "avgpool"
        self.kernel_size = 7

        # Runtime state, reset at every prefill.
        self.prompt_len = 0
        self.num_comp = 0
        self.cached_recent = None

    def reset(self, prompt_len):
        self.prompt_len = prompt_len
        self.num_comp = 0
        self.cached_recent = None

    def cache_recent(self, current_query_states):
        if self.cached_recent is None:
            self.cached_recent = current_query_states
        else:
            self.cached_recent = torch.cat([self.cached_recent, current_query_states], dim=-2)

    # ── Block-based adaptive selector ──

    def _compute_block_signals(self, block_q_norm, block_idx, num_blocks):
        """
        Compute recency and diversity signals for a single block.
        block_q_norm: [block_size, head_dim] (L2-normalized)
        Returns: (recency, diversity)
        """
        # Recency: exponential decay with half-life
        decay = math.log(2) / max(self.half_life, 1e-6)
        recency = math.exp(-decay * (num_blocks - 1 - block_idx))

        # Diversity: mean pairwise cosine distance within block
        if block_q_norm.shape[0] < 2:
            diversity = 0.0
        else:
            cos_sim = torch.mm(block_q_norm, block_q_norm.t())
            mask = ~torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            diversity = max((1 - cos_sim[mask]).mean().item(), 0.0)

        return recency, diversity

    @staticmethod
    def _normalize_block_signal(signal):
        total = signal.sum()
        if total.item() <= 0:
            return torch.zeros_like(signal)
        return signal / total

    def _combine_block_signals(self, recencies, diversities):
        recency_weights = self._normalize_block_signal(recencies)
        diversity_weights = self._normalize_block_signal(diversities)
        return self.alpha * recency_weights + (1 - self.alpha) * diversity_weights

    def _allocate_quotas_proportional(self, utilities, budget):
        """
        Largest-remainder allocation of `budget` slots across blocks,
        proportional to `utilities`. Returns: list of ints summing to budget.
        """
        num_blocks = utilities.shape[0]
        total_util = utilities.sum()

        if total_util == 0:
            quotas = [0] * num_blocks
            for i in range(budget):
                quotas[-(i % num_blocks + 1)] += 1
            return quotas

        raw = utilities / total_util * budget
        quotas = [int(r.item()) for r in raw.floor()]

        # Distribute remainder to blocks with highest fractional parts
        remainder = budget - sum(quotas)
        fracs = [(raw[i] - quotas[i]).item() for i in range(num_blocks)]
        for _ in range(remainder):
            max_idx = max(range(num_blocks), key=lambda i: fracs[i])
            quotas[max_idx] += 1
            fracs[max_idx] = -1.0
        return quotas

    def _allocate_quotas(self, utilities, K):
        """
        Allocate K budget across blocks, guaranteeing >=1 slot per block
        whenever K allows it, and distributing the remaining budget
        proportional to utilities (largest-remainder).
        utilities: [num_blocks] tensor
        Returns: list of ints summing to K
        """
        num_blocks = utilities.shape[0]

        if K < num_blocks:
            # Not enough budget to reserve 1 per block; fall back to a
            # pure proportional allocation over all K slots.
            return self._allocate_quotas_proportional(utilities, K)

        remaining = K - num_blocks
        if remaining == 0:
            return [1] * num_blocks

        extra_quotas = self._allocate_quotas_proportional(utilities, remaining)
        return [1 + q for q in extra_quotas]

    def _within_block_select(self, block_q_norm, quota):
        """
        Select quota representative queries within a block.
        block_q_norm: [block_size, head_dim] (L2-normalized)
        Returns: list of local indices
        """
        block_size = block_q_norm.shape[0]
        if quota >= block_size:
            return list(range(block_size))

        cos_sim = torch.mm(block_q_norm, block_q_norm.t())

        # First seed: medoid (most central query)
        centrality = cos_sim.mean(dim=1)
        seed = centrality.argmax().item()
        indices = [seed]

        if quota == 1:
            return indices

        # Remaining: pure farthest point selection
        min_dist = 1 - cos_sim[seed]
        min_dist[seed] = -1.0

        for _ in range(quota - 1):
            next_idx = min_dist.argmax().item()
            indices.append(next_idx)
            new_dist = 1 - cos_sim[next_idx]
            min_dist = torch.minimum(min_dist, new_dist)
            min_dist[next_idx] = -1.0

        return indices

    def _block_select(self, queries, K):
        """
        Block-based adaptive selector.
        queries: [bsz, num_heads, N, head_dim]
        Returns: (selectors [bsz, num_heads, K, head_dim], selected_indices [bsz, K])
        """
        bsz, num_heads, N, head_dim = queries.shape

        if N <= K:
            indices = torch.arange(N, device=queries.device).unsqueeze(0).expand(bsz, -1)
            return queries, indices

        q_avg = queries.mean(dim=1)          # [bsz, N, head_dim]
        q_norm = F.normalize(q_avg, dim=-1)  # [bsz, N, head_dim]

        num_full_blocks = N // self.block_size
        remainder = N % self.block_size

        selected_indices = []
        for b in range(bsz):
            block_ranges = []
            block_recencies = []
            block_diversities = []

            for m in range(num_full_blocks):
                start = m * self.block_size
                end = start + self.block_size
                block_ranges.append((start, end))
                recency, diversity = self._compute_block_signals(
                    q_norm[b, start:end],
                    m,
                    num_full_blocks + (1 if remainder > 0 else 0),
                )
                block_recencies.append(recency)
                block_diversities.append(diversity)

            if remainder > 0:
                start = num_full_blocks * self.block_size
                end = N
                block_ranges.append((start, end))
                total_blocks = num_full_blocks + 1
                recency, diversity = self._compute_block_signals(
                    q_norm[b, start:end], num_full_blocks, total_blocks
                )
                block_recencies.append(recency)
                block_diversities.append(diversity)

            recencies = torch.tensor(block_recencies, device=queries.device)
            diversities = torch.tensor(block_diversities, device=queries.device)
            utilities = self._combine_block_signals(recencies, diversities)
            quotas = self._allocate_quotas(utilities, K)

            batch_indices = []
            for m, (start, end) in enumerate(block_ranges):
                if quotas[m] == 0:
                    continue
                block_q = q_norm[b, start:end]
                actual_quota = min(quotas[m], end - start)
                local_idx = self._within_block_select(block_q, actual_quota)
                batch_indices.extend([start + li for li in local_idx])

            # Ensure exactly K indices
            if len(batch_indices) > K:
                batch_indices = batch_indices[:K]
            while len(batch_indices) < K:
                for idx in range(N - 1, -1, -1):  # recent first
                    if idx not in batch_indices:
                        batch_indices.append(idx)
                        if len(batch_indices) == K:
                            break

            batch_indices = sorted(batch_indices)
            selected_indices.append(batch_indices)

        # Log (once per problem, first compression only)
        if self.layer_idx == 0 and self.num_comp == 0:
            print(f"[Block Selector] N={N}, K={K}, blocks={len(block_ranges)}, "
                  f"quota_dist=[min={min(quotas)}, max={max(quotas)}, nonzero={sum(1 for q in quotas if q > 0)}]", flush=True)

        idx_tensor = torch.tensor(selected_indices, device=queries.device)  # [bsz, K]
        idx_expand = idx_tensor.unsqueeze(1).unsqueeze(-1).expand(bsz, num_heads, K, head_dim)
        selectors = queries.gather(dim=2, index=idx_expand)

        return selectors, idx_tensor

    def _aggregate_and_pool_attention(self, attn_weights):
        """Aggregate attention into one importance score per candidate KV."""
        if self.scoring == 'attention_sum':
            scores = attn_weights.sum(dim=-2)
            head_reduce = torch.mean
        elif self.scoring == 'attention_max':
            # BeaconKV scoring: max over selector queries, then max over
            # attention heads that share the same KV head.
            scores = attn_weights.amax(dim=-2)
            head_reduce = torch.amax
        else:
            raise ValueError(f"Unsupported attention scoring policy: {self.scoring}")

        if self.aggregation == 'all':
            scores = scores.view(scores.shape[0], -1, 1, scores.shape[-1])
            scores = head_reduce(scores, dim=-3)
        elif self.aggregation == 'group':
            scores = scores.view(
                scores.shape[0],
                -1,
                self.num_key_value_groups,
                scores.shape[-1],
            )
            scores = head_reduce(scores, dim=-2)
        elif self.aggregation != 'none':
            raise ValueError(f"Unsupported head aggregation policy: {self.aggregation}")

        if self.scoring == 'attention_max':
            # BeaconKV does not apply local sequence pooling after the max
            # reductions.
            return scores

        if self.pooling == 'avgpool':
            return F.avg_pool1d(
                scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        if self.pooling == 'maxpool':
            return F.max_pool1d(
                scores,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        raise ValueError(f"Unsupported sequence pooling policy: {self.pooling}")

    def _compute_pooled_scores(self, selectors, key_states, head_dim):
        """
        Compute per-query attention scores, aggregate heads, and pool.
        selectors: [bsz, num_heads, q_len, head_dim]
        key_states: [bsz, num_heads, k_len, head_dim] (already GQA-expanded)
        Returns: [bsz, 1, k_len] pooled scores (aggregation='all')
                 or [bsz, num_kv_heads, k_len] (aggregation='group')
        """
        attn_weights = torch.matmul(selectors, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(
            attn_weights[:, :, :, self.prompt_len:-self.R],
            dim=-1, dtype=torch.float32
        ).to(selectors.dtype)
        return self._aggregate_and_pool_attention(attn_weights)

    def compress_kv_rpc(self, origin_key_states, origin_value_states, query_states):
        """RPC: score past generated tokens with the last R queries."""
        if self.cached_recent is not None:
            selectors = torch.cat([self.cached_recent, query_states], dim=-2)
        else:
            selectors = query_states
        self.cached_recent = None

        # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)

        head_dim = selectors.shape[-1]
        total_keep = (self.num_comp + 1) * self.T

        attn_cache = self._compute_pooled_scores(selectors, key_states, head_dim)
        selected_rel_indices = attn_cache.topk(total_keep, dim=-1, largest=True).indices.sort(dim=-1).values
        indices = selected_rel_indices.unsqueeze(-1).expand(-1, origin_key_states.size(1), -1, head_dim)

        k_prompt = origin_key_states[:, :, :self.prompt_len, :]
        v_prompt = origin_value_states[:, :, :self.prompt_len, :]

        k_past_compress = origin_key_states[:, :, self.prompt_len:-self.R, :].gather(dim=2, index=indices)
        v_past_compress = origin_value_states[:, :, self.prompt_len:-self.R, :].gather(dim=2, index=indices)

        k_cur = origin_key_states[:, :, -self.R:, :]
        v_cur = origin_value_states[:, :, -self.R:, :]

        key_states = torch.cat([k_prompt, k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_prompt, v_past_compress, v_cur], dim=2)
        return key_states, value_states

    def compress_kv_rewindkv(self, origin_key_states, origin_value_states, query_states):
        """RewindKV: score all generated tokens with K block-selected queries."""

        # Step 1: Build selector set via block selection
        all_queries = torch.cat([self.cached_recent, query_states], dim=-2)
        selectors, selected_indices = self._block_select(all_queries, self.K)
        # selected_indices: [bsz, K] -- indices into all_queries
        self.cached_recent = None

        # Step 2: GQA support
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)

        head_dim = selectors.shape[-1]
        total_keep = (self.num_comp + 1) * self.T + self.K

        # Step 3: Score all generated tokens
        gen_keys = key_states[:, :, self.prompt_len:, :]
        gen_len = gen_keys.shape[2]

        attn_weights = torch.matmul(selectors, gen_keys.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(selectors.dtype)
        scores = self._aggregate_and_pool_attention(attn_weights)

        # Step 4: Always keep the selector tokens (score = inf)
        num_new = all_queries.shape[2]
        offset = gen_len - num_new
        preserve_pos = selected_indices + offset  # [bsz, K]
        scatter_index = preserve_pos.unsqueeze(1).expand(-1, scores.shape[1], -1)
        scores.scatter_(-1, scatter_index, float('inf'))

        # Step 5: Top-k selection and reconstruction
        selected_rel_indices = scores.topk(total_keep, dim=-1, largest=True).indices.sort(dim=-1).values
        indices = selected_rel_indices.unsqueeze(-1).expand(-1, origin_key_states.size(1), -1, head_dim)

        k_prompt = origin_key_states[:, :, :self.prompt_len, :]
        v_prompt = origin_value_states[:, :, :self.prompt_len, :]

        k_compressed = origin_key_states[:, :, self.prompt_len:, :].gather(dim=2, index=indices)
        v_compressed = origin_value_states[:, :, self.prompt_len:, :].gather(dim=2, index=indices)

        key_states = torch.cat([k_prompt, k_compressed], dim=2)
        value_states = torch.cat([v_prompt, v_compressed], dim=2)
        return key_states, value_states

    def compress_kv(self, key_states, value_states, query_states):
        if self.method == "rewindkv":
            return self.compress_kv_rewindkv(key_states, value_states, query_states)
        return self.compress_kv_rpc(key_states, value_states, query_states)


def init_kv_cluster(attn):
    if not hasattr(attn, "kv_cluster"):
        attn.kv_cluster = KVCluster(
            layer_idx=attn.layer_idx,
            num_key_value_groups=attn.config.num_attention_heads // attn.config.num_key_value_heads,
        )

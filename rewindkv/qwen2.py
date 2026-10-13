from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    Qwen2FlashAttention2,
)
from transformers.utils import logging

from rewindkv.cluster import init_kv_cluster

logger = logging.get_logger(__name__)


class Qwen2CompressedAttention(Qwen2FlashAttention2):
    """Qwen2 flash-attention layer whose KV cache is periodically compressed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_kv_cluster(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            # (kept from transformers' Qwen2FlashAttention2 so that the attention mask matches it)
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            kv_seq_len = key_states.shape[-2] + cache_position[0]
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            cluster = self.kv_cluster

            if q_len == 1:
                # Decoding. Number of generated tokens added since the last compression,
                # excluding the R (rpc) / K (rewindkv) tokens kept from the selectors.
                # (past_key_value.get_seq_length cannot be used: the cache is compressed.)
                target_length = (
                    past_key_value.key_cache[self.layer_idx].size(-2)
                    - cluster.prompt_len
                    - cluster.num_comp * cluster.T
                    - cluster.R
                )

                if cluster.method == "rewindkv" and target_length > 0:
                    # every query since the last compression is a selector candidate
                    cluster.cache_recent(query_states)
                elif cluster.method == "rpc" and target_length > cluster.P - cluster.R:
                    # the last R queries are the selectors
                    cluster.cache_recent(query_states)

                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                if target_length == cluster.P - 1:
                    key_states_compress, value_states_compress = cluster.compress_kv(key_states, value_states, query_states)
                    # attend with the full cache now, expose the compressed cache from the next step
                    past_key_value.key_cache[self.layer_idx] = key_states_compress
                    past_key_value.value_cache[self.layer_idx] = value_states_compress
                    cluster.num_comp += 1
            else:
                # Prefill: the prompt is never compressed.
                past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                cluster.reset(prompt_len=key_states.size(-2))

        # flash-attention handles GQA, no repeat_kv needed
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reshape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask[:, :key_states.shape[1]] if attention_mask is not None else attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value

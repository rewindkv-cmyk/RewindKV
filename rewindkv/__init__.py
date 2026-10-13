import warnings
from importlib.metadata import version

from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES

from rewindkv.cluster import METHODS, set_compression_config
from rewindkv.qwen2 import Qwen2CompressedAttention

SUPPORTED_TRANSFORMERS = "4.45"


def enable_kv_compression():
    """Swap the Qwen2 flash-attention layer for the compressed one.

    Call before ``AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")``,
    then configure the method with ``set_compression_config``.
    """
    transformers_version = version("transformers")
    if not transformers_version.startswith(SUPPORTED_TRANSFORMERS):
        warnings.warn(f"transformers {transformers_version} may be incompatible; "
                      f"this code is tested with transformers {SUPPORTED_TRANSFORMERS}.")
    QWEN2_ATTENTION_CLASSES["flash_attention_2"] = Qwen2CompressedAttention


__all__ = ["METHODS", "enable_kv_compression", "set_compression_config"]

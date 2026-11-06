"""B-LoRA utilities.

Helpers for identifying, filtering, scaling and loading the two attention
blocks ("B-LoRA" blocks) that the B-LoRA method trains inside the SDXL UNet:

    content block: ``unet.up_blocks.0.attentions.0``
    style   block: ``unet.up_blocks.0.attentions.1``

Updated for modern diffusers (>= 0.27, PEFT-based LoRA) and Python 3.13.
"""
from pathlib import Path

import torch

BLOCKS = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1'],
}

# File names that B-LoRA / diffusers checkpoints may use inside a directory.
_LORA_WEIGHT_FILES = (
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
    "lora_weights.safetensors",
    "lora_weights.bin",
)


def is_belong_to_blocks(key, blocks):
    """Return True if ``key`` belongs to one of the given ``blocks``."""
    return any(g in key for g in blocks)


def filter_lora(state_dict, blocks_):
    """Keep only the state-dict entries that belong to the given blocks."""
    return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}


def scale_lora(state_dict, alpha):
    """Scale the LoRA weights of a (filtered) state dict by ``alpha``.

    Only the low-rank "down" / "A" factor is scaled. Scaling both factors
    (A *and* B) would scale the effective weight update ``B @ A`` by
    ``alpha ** 2`` instead of ``alpha``.

    Supported key formats:
      - ``...lora.down.weight``  (diffusers format, what we save)
      - ``...lora_A.weight``     (raw PEFT format)
      - ``...lora.weight``       (very old combined single-tensor format)
    """
    scaled = {}
    for k, v in state_dict.items():
        if k.endswith("lora.down.weight") or k.endswith("lora_A.weight") or k.endswith("lora.weight"):
            v = v * alpha
        scaled[k] = v
    return scaled


def normalize_lora_keys(state_dict):
    """Normalize LoRA checkpoint keys so modern diffusers can consume them.

    Legacy B-LoRA checkpoints (e.g. ``lora-library/B-LoRA-*`` on the Hugging
    Face Hub) were saved with a doubled prefix, ``unet.unet.up_blocks.0...``.
    Modern diffusers strips only a single ``unet.`` prefix, which leaves a
    stray ``unet.`` that breaks module matching. This collapses the doubled
    prefix (and any further duplication) back to ``unet.``.
    """
    normalized = {}
    for k, v in state_dict.items():
        while k.startswith("unet.unet."):
            k = k[len("unet."):]
        normalized[k] = v
    return normalized


def load_lora_state_dict(path_or_repo_id):
    """Load a LoRA state dict from a local file, a local directory, or a
    Hugging Face Hub repo id (e.g. ``lora-library/B-LoRA-teddybear``).

    Returns a state dict with normalized keys (see :func:`normalize_lora_keys`).
    """
    candidate = Path(path_or_repo_id)

    if candidate.is_dir():
        for name in _LORA_WEIGHT_FILES:
            f = candidate / name
            if f.exists():
                candidate = f
                break
        else:
            raise FileNotFoundError(
                f"No LoRA weights file found in '{candidate}'. "
                f"Expected one of: {', '.join(_LORA_WEIGHT_FILES)}"
            )
    elif not candidate.exists():
        # Treat as a Hugging Face Hub repo id.
        from huggingface_hub import hf_hub_download

        candidate = None
        for name in _LORA_WEIGHT_FILES:
            try:
                candidate = Path(hf_hub_download(repo_id=str(path_or_repo_id), filename=name))
                break
            except Exception:
                continue
        if candidate is None:
            raise FileNotFoundError(
                f"Could not find LoRA weights for '{path_or_repo_id}' "
                f"(looked for: {', '.join(_LORA_WEIGHT_FILES)})"
            )

    if candidate.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(candidate))
    else:
        state_dict = torch.load(str(candidate), map_location="cpu", weights_only=True)

    return normalize_lora_keys(state_dict)


def get_target_modules(unet, blocks=None):
    """Return the LoRA target module paths for the given blocks.

    Enumerates the UNet attention processors that belong to ``blocks`` and
    builds the full module paths of their ``to_q`` / ``to_k`` / ``to_v`` /
    ``to_out.0`` projections. The resulting list is used as PEFT
    ``LoraConfig.target_modules`` and covers exactly the same modules as the
    original (pre-PEFT) B-LoRA implementation.
    """
    if not blocks:
        blocks = [('.').join(blk.split('.')[1:]) for blk in BLOCKS['content'] + BLOCKS['style']]

    attns = [attn_processor_name.rsplit('.', 1)[0] for attn_processor_name, _ in unet.attn_processors.items()
             if is_belong_to_blocks(attn_processor_name, blocks)]

    return [f'{attn}.{mat}' for mat in ["to_k", "to_q", "to_v", "to_out.0"] for attn in attns]

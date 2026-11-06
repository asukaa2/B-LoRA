"""B-LoRA CPU smoke tests.

These tests run on CPU without any model downloads (a miniature SDXL-like
UNet is constructed locally), so they execute quickly and work offline
(except the optional Hub test, enabled with BLORA_TEST_HF=1).

Run with either:
    python -m pytest tests/ -v
    python tests/test_cpu_smoke.py
"""
import ast
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blora_utils import (  # noqa: E402
    BLOCKS,
    filter_lora,
    get_target_modules,
    load_lora_state_dict,
    normalize_lora_keys,
    scale_lora,
)

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel  # noqa: E402
from diffusers.utils import convert_state_dict_to_diffusers  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict  # noqa: E402
import re  # noqa: E402

RANK = 64
# Miniature SDXL: same block layout as the real model (up_blocks.0 is a
# CrossAttnUpBlock2D with 2 attentions x 10 transformer layers) but tiny dims.
UNET_KWARGS = dict(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    block_out_channels=(32, 64, 128),
    transformer_layers_per_block=(1, 2, 10),
    layers_per_block=2,
    cross_attention_dim=64,
    attention_head_dim=(4, 8, 16),
)
EXPECTED_MODULES_PER_BLOCK = 10 * 2 * 4  # layers x attentions x (q,k,v,out)


def make_unet(seed=0):
    torch.manual_seed(seed)
    return UNet2DConditionModel(**UNET_KWARGS)


def make_lora_config(unet):
    return LoraConfig(
        r=RANK,
        lora_alpha=RANK,  # scaling = 1.0, matches the original implementation
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
        target_modules=get_target_modules(unet),
    )


def test_target_modules_match_original_blocks():
    """PEFT must wrap exactly the two B-LoRA blocks, nothing else."""
    unet = make_unet()
    targets = get_target_modules(unet)
    assert len(targets) == 2 * EXPECTED_MODULES_PER_BLOCK
    for t in targets:
        assert t.startswith(("up_blocks.0.attentions.0.", "up_blocks.0.attentions.1."))
        assert t.split(".")[-1] in {"to_q", "to_k", "to_v"} or t.endswith("to_out.0")
    unet.add_adapter(make_lora_config(unet))
    wrapped = {n.split(".lora_A")[0] for n, _ in unet.named_parameters() if ".lora_A." in n}
    assert len(wrapped) == 2 * EXPECTED_MODULES_PER_BLOCK
    for w in wrapped:
        assert w.startswith(("up_blocks.0.attentions.0.", "up_blocks.0.attentions.1.")), w
    # nothing outside the two blocks may be trainable
    outside = [n for n, p in unet.named_parameters()
               if p.requires_grad and not n.startswith(("up_blocks.0.attentions.0", "up_blocks.0.attentions.1"))]
    assert not outside, outside


def _lora_param_map(model):
    """Map (module_path, factor) -> tensor, ignoring the adapter-name segment."""
    out = {}
    for n, p in model.named_parameters():
        m = re.search(r"\.lora_([AB])\.[^.]+\.weight$", n)
        if m:
            out[(n.split(".lora_")[0], m.group(1))] = p
    return out


def test_save_and_load_roundtrip():
    """save_lora_weights -> load -> load_lora_into_unet must be lossless."""
    unet = make_unet()
    config = make_lora_config(unet)
    unet.add_adapter(config)
    with torch.no_grad():
        for p in unet.parameters():
            if p.requires_grad:
                p.add_(0.05)

    diff_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    with tempfile.TemporaryDirectory() as tmp:
        StableDiffusionXLPipeline.save_lora_weights(tmp, unet_lora_layers=diff_sd)
        # exercise the full load path used by inference.py
        sd = load_lora_state_dict(tmp)
    assert len(sd) == 2 * EXPECTED_MODULES_PER_BLOCK * 2
    assert all(k.startswith("unet.") for k in sd)

    fresh = make_unet(seed=1)
    fresh.add_adapter(config)
    StableDiffusionXLPipeline.load_lora_into_unet(sd, network_alphas=None, unet=fresh)
    ref, got = _lora_param_map(unet), _lora_param_map(fresh)
    assert set(ref) == set(got)
    for n in ref:
        assert torch.allclose(ref[n], got[n], atol=1e-6), n


def test_legacy_doubled_prefix_checkpoint():
    """Legacy checkpoints (unet.unet.* keys, lora.down/up) must load."""
    unet = make_unet()
    config = make_lora_config(unet)
    unet.add_adapter(config)
    with torch.no_grad():
        for p in unet.parameters():
            if p.requires_grad:
                p.add_(0.05)
    diff_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    # emulate the legacy serialization: doubled prefix, saved by diffusers 0.25
    legacy_sd = {f"unet.{k}": v for k, v in diff_sd.items()}  # unet.up_blocks...
    legacy_sd = {f"unet.{k}": v for k, v in legacy_sd.items()}  # unet.unet.up_blocks...

    with tempfile.TemporaryDirectory() as tmp:
        from safetensors.torch import save_file
        save_file(legacy_sd, str(Path(tmp) / "pytorch_lora_weights.safetensors"))
        sd = load_lora_state_dict(tmp)  # normalization happens inside
    assert all(not k.startswith("unet.unet.") for k in sd)

    fresh = make_unet(seed=2)
    fresh.add_adapter(config)
    StableDiffusionXLPipeline.load_lora_into_unet(sd, network_alphas=None, unet=fresh)
    ref, got = _lora_param_map(unet), _lora_param_map(fresh)
    assert set(ref) == set(got)
    for n in ref:
        assert torch.allclose(ref[n], got[n], atol=1e-6), n


def test_scale_lora_alpha_semantics():
    """alpha must scale B@A once: only the A/down factor may be multiplied."""
    torch.manual_seed(3)
    key_a = "unet.up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight"
    key_b = "unet.up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight"
    A = torch.randn(8, 16)
    B = torch.randn(16, 8)
    sd = {key_a: A.clone(), key_b: B.clone()}
    scaled = scale_lora(sd, 0.7)
    assert torch.allclose(scaled[key_a], A * 0.7)
    assert torch.allclose(scaled[key_b], B)  # B untouched!
    # effective update is alpha * (B @ A), not alpha^2 * (B @ A)
    effective = scaled[key_b] @ scaled[key_a]
    assert torch.allclose(effective, 0.7 * (B @ A), atol=1e-6)

    # PEFT-format keys
    sd2 = {key_a.replace("lora.down.weight", "lora_A.weight"): A.clone(),
           key_b.replace("lora.up.weight", "lora_B.weight"): B.clone()}
    scaled2 = scale_lora(sd2, 1.3)
    assert torch.allclose(scaled2[key_a.replace("lora.down.weight", "lora_A.weight")], A * 1.3)
    assert torch.allclose(scaled2[key_b.replace("lora.up.weight", "lora_B.weight")], B)


def test_filter_lora_blocks():
    unet = make_unet()
    targets = get_target_modules(unet)
    sd = {}
    for t in targets:
        sd[f"unet.{t}.lora.down.weight"] = torch.randn(4, 4)
    content = filter_lora(sd, BLOCKS["content"])
    style = filter_lora(sd, BLOCKS["style"])
    assert all("attentions.0." in k for k in content)
    assert all("attentions.1." in k for k in style)
    assert len(content) == EXPECTED_MODULES_PER_BLOCK
    assert len(style) == EXPECTED_MODULES_PER_BLOCK
    assert len({**content, **style}) == len(sd)  # no overlap when merged


def test_normalize_lora_keys_idempotent():
    sd = {"unet.unet.up_blocks.0.attentions.0.to_q.lora.down.weight": torch.zeros(1),
          "unet.up_blocks.0.attentions.1.to_q.lora.down.weight": torch.zeros(1)}
    once = normalize_lora_keys(sd)
    twice = normalize_lora_keys(once)
    assert all(k.startswith("unet.up_blocks") for k in once)
    assert once == twice


def test_no_removed_python_313_apis():
    """logger.warn was removed in Python 3.13 - it must not be used."""
    repo = Path(__file__).resolve().parent.parent
    for py in repo.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "warn"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"):
                raise AssertionError(f"{py.name}: logger.warn removed in Python 3.13, use logger.warning")


def test_train_script_imports():
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_dreambooth_b_lora_sdxl", repo / "train_dreambooth_b-lora_sdxl.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main") and hasattr(module, "parse_args")
    assert hasattr(module, "TEXT_ENCODER_TARGET_MODULES")


def test_hf_hub_checkpoint_loads():
    """Optional (BLORA_TEST_HF=1): load a real legacy B-LoRA from the Hub and
    verify normalization + block filtering on the real 320-key checkpoint."""
    if os.environ.get("BLORA_TEST_HF") != "1":
        print("  (skipped: set BLORA_TEST_HF=1 to test Hub download)")
        return
    sd = load_lora_state_dict("lora-library/B-LoRA-teddybear")
    assert len(sd) == 320
    assert all(k.startswith("unet.up_blocks.0.") for k in sd)
    content = filter_lora(sd, BLOCKS["content"])
    style = filter_lora(sd, BLOCKS["style"])
    assert len(content) == 160 and len(style) == 160


TESTS = [
    test_target_modules_match_original_blocks,
    test_save_and_load_roundtrip,
    test_legacy_doubled_prefix_checkpoint,
    test_scale_lora_alpha_semantics,
    test_filter_lora_blocks,
    test_normalize_lora_keys_idempotent,
    test_no_removed_python_313_apis,
    test_train_script_imports,
    test_hf_hub_checkpoint_loads,
]

if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    sys.exit(1 if failures else 0)

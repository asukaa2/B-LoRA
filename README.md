# Implicit Style-Content Separation using B-LoRA
<a href="https://B-LoRA.github.io/B-LoRA/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a> [![arXiv](https://img.shields.io/badge/arXiv-2403.14572-b31b1b.svg)](https://arxiv.org/abs/2403.14572)
[![Open In Colab — Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asukaa2/B-LoRA/blob/main/B_LoRA_inference.ipynb) [![Open In Colab — Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asukaa2/B-LoRA/blob/main/B_LoRA_training.ipynb) [![HuggingFace demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Yardenfren/B-LoRA)

![Teaser Image](docs/teaser_blora.png)

This repository contains the official implementation of the B-LoRA method, which enables implicit style-content separation of a single input image for various image stylization tasks. B-LoRA leverages the power of Stable Diffusion XL (SDXL) and Low-Rank Adaptation (LoRA) to disentangle the style and content components of an image, facilitating applications such as image style transfer, text-based image stylization, and consistent style generation.

| Colab demo | What it does |
|---|---|
| **Training** (`B_LoRA_training.ipynb`) | Train a B-LoRA from a single image on a free T4 GPU (~30–40 min in quick mode), then run content/style separation demos with the result |
| **Inference** (`B_LoRA_inference.ipynb`) | Use pretrained B-LoRAs (`lora-library/B-LoRA-*`) for image stylization, text-based stylization and consistent style generation |

## 🆕 Modern diffusers / PEFT + Python 3.13 update

The original training script targeted `diffusers==0.25.0` and injected LoRA layers through
`Attention.set_lora_layer(...)`, an API that **no longer exists** in current diffusers
(and whose dependencies — `Pillow 9.x`, `bitsandbytes 0.36` — cannot be installed on
Python ≥ 3.13). The repository has been modernized:

* **Training** now uses the PEFT-based LoRA backend. A `peft.LoraConfig` is built with
  dynamically computed `target_modules` (`blora_utils.get_target_modules`), which targets
  exactly the same modules as the original implementation: the `to_q/to_k/to_v/to_out.0`
  projections of every attention layer inside `unet.up_blocks.0.attentions.{0,1}`.
  `lora_alpha` defaults to `rank` (effective scaling 1.0), matching the original scaling.
* **Inference** no longer relies on the removed `pipeline.lora_state_dict()`;
  `blora_utils.load_lora_state_dict()` loads checkpoints from a local path/directory or a
  HF Hub repo id and normalizes legacy key formats (old `unet.unet.*`-prefixed
  checkpoints from the original release load unchanged).
* **`--train_text_encoder`** is implemented with PEFT text-encoder adapters (the original
  `LoraLoaderMixin._modify_text_encoder` monkey-patching was removed upstream).
* **Python 3.13 fixes**: `logger.warn` (removed in Python 3.13) → `logger.warning`;
  all dependencies pinned to versions with Python 3.13 wheels; `torch.load` uses
  `weights_only=True`.
* **New `tests/` suite** — CPU-only smoke tests (miniature SDXL UNet) that verify block
  targeting, checkpoint round-trips, legacy checkpoint compatibility and alpha scaling.
  Run with `python tests/test_cpu_smoke.py` or `pytest tests/`.

> Historical note (21.5.2024): the original authors reported convergence issues with
> newer diffusers + PEFT versions at that time. The current script uses the PEFT APIs
> the way current `diffusers` training scripts do, with the same block targeting, rank
> (64) and scaling (1.0) as the paper. If you need to reproduce the exact paper-era
> environment, use `git checkout` of the pre-update history with Python 3.11.

## Getting Started

### Prerequisites
- Python 3.10 – 3.13 (tested on 3.13.5)
- PyTorch 2.6+ (2.6 is the first release with official Python 3.13 wheels)
- A CUDA GPU for training/inference (the test suite runs on CPU)
- Other dependencies (specified in `requirements.txt`)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/asukaa2/B-LoRA.git
   cd B-LoRA
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   For GPU training install the CUDA build of PyTorch first, e.g.:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

3. (Optional) run the CPU smoke tests:
   ```
   python tests/test_cpu_smoke.py
   ```

### Usage

1. **Training B-LoRAs**

   To train the B-LoRAs for a given input image, run:
   ```
   accelerate launch train_dreambooth_b-lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --instance_data_dir="<path/to/example_images>" \
    --output_dir="<path/to/output_dir>" \
    --instance_prompt="<prompt>" \
    --resolution=1024 \
    --rank=64 \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --seed="0" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision="fp16"
      ```
This will optimize the B-LoRA weights for the content and style and store them in  `output_dir`.
Parameters that need to replace  `instance_data_dir`, `output_dir`, `instance_prompt` (in our paper we use `A [v]`)

> 💡 No GPU handy? Open the training Colab: it runs the same command in *quick* mode
> (768 px, 300 steps) on a free T4, with the paper settings available as an option.

![Apps Image](docs/apps_method1.png)

2. **Inference**   

   For image stylization based on a reference style image (1), run:
   ```
   python inference.py --prompt="A <c> in <s> style" --content_B_LoRA="<path/to/content_B-LoRA>" --style_B_LoRA="<path/to/style_B-LoRA>" --output_path="<path/to/output_dir>"
   ```
   This will generate new images with the content of the first B-LoRA and the style of the second B-LoRA.
   Note that you need to replace `c` and `s` in the prompt according to the optimization prompt.

   For text-based image stylization (2), run:
   ```
   python inference.py --prompt="A <c> made of gold" --content_B_LoRA="<path/to/content_B-LoRA>" --output_path="<path/to/output_dir>"
   ```
   This will generate new images with the content of the given B-LoRA and the style specified by the text prompt.

   For consistent style generation (3), run:
   ```
   python inference.py --prompt="A backpack in <s> style" --style_B_LoRA="<path/to/style_B-LoRA>" --output_path="<path/to/output_dir>"
   ```
   This will generate new images with the specified content and the style of the given B-LoRA.

   `--content_B_LoRA` / `--style_B_LoRA` accept a directory, a checkpoint file
   (`.safetensors` / `.bin`) or a HF Hub repo id such as `lora-library/B-LoRA-teddybear`.

   Several additional parameters that you can set in the `inference.py` file include:
   1. `--content_alpha`, `--style_alpha` for controlling the strength of the adapters.
   2. `--num_images_per_prompt` for specifying the number of output images.
   3. `--num_inference_steps`, `--guidance_scale`, `--negative_prompt` and `--seed`.

   (For a111 and comfy see this [issue](https://github.com/yardenfren1996/B-LoRA/issues/7))

## Repository layout

| Path | Description |
|---|---|
| `train_dreambooth_b-lora_sdxl.py` | Training script (PEFT-based, SDXL) |
| `inference.py` | CLI inference (content/style mixing, alpha scaling) |
| `blora_utils.py` | Block constants, filter/scale/normalize/load helpers, target-module builder |
| `B_LoRA_training.ipynb` | **Colab training demo** (quick + paper modes) |
| `B_LoRA_inference.ipynb` | Colab inference demo (pretrained B-LoRAs) |
| `tests/test_cpu_smoke.py` | CPU smoke tests (no GPU / no downloads required) |

## Changelog (Python 3.13 / modern diffusers update)

* `requirements.txt`: `diffusers>=0.32`, `peft>=0.14`, `torch>=2.6`, `Pillow>=11.0`,
  `bitsandbytes>=0.45`, `safetensors>=0.4.5` — all installable on Python 3.10–3.13.
* Training: manual `LoRALinearLayer` injection → `peft.LoraConfig` with dynamic
  `target_modules`; saving via `get_peft_model_state_dict` +
  `convert_state_dict_to_diffusers`; checkpoint hooks updated; `DreamBoothDataset`
  no longer reads a module-level global `args`.
* Inference: removed `pipeline.lora_state_dict()` usage; state dicts are loaded and
  normalized in `blora_utils`, then attached with `pipeline.load_lora_weights(sd, adapter_name="blora")`.
* `blora_utils.scale_lora` now scales only the `lora.down` / `lora_A` factor, so
  `alpha` scales the effective update `B @ A` by exactly `alpha` (scaling both factors
  would square it).
* `blora_utils.normalize_lora_keys` + `load_lora_state_dict` added: legacy
  `unet.unet.*` checkpoints (e.g. `lora-library/B-LoRA-*`) load correctly.
* Both Colab notebooks updated to the new APIs; a training demo notebook was added.

## Citation

If you use B-LoRA in your research, please cite the following paper:

```bibtex
@misc{frenkel2024implicit,
      title={Implicit Style-Content Separation using B-LoRA}, 
      author={Yarden Frenkel and Yael Vinker and Ariel Shamir and Daniel Cohen-Or},
      year={2024},
      eprint={2403.14572},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, please feel free to open an issue or contact the authors at [yardenfren@gmail.com](mailto:yardenfren@gmail.com).

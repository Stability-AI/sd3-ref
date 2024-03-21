# Stable Diffusion 3 Micro-Reference Implementation

Inference-only tiny reference implementation of SD3.

Contains code for the text encoders (OpenAI CLIP-L/14, OpenCLIP bigG, Google T5-XXL) (these models are all public), the VAE Decoder (similar to previous SD models, but 16-channels and no postquantconv step), and the core MM-DiT (entirely new).

Everything you need to inference SD3 excluding the weights files.

### Install

```sh
# Note: on windows use "python" not "python3"
python3 -s -m venv venv
source ./venv/bin/activate
# or on windows: venv/scripts/activate
python3 -s -m pip install -r requirements.txt
```

### Test Usage

```sh
# Generate a cat on ref model with default settings
python3 -s sd3_infer.py
# Generate a 1024 cat on SD3-8B
python3 -s sd3_infer.py --width 1024 --height 1024 --shift 3 --model models/sd3_8b_beta.safetensors --prompt "cute wallpaper art of a cat"
# Or for parameter listing
python3 -s sd3_infer.py --help
```

Images will be output to `output.png` by default

### File Guide

- `sd3_infer.py` - entry point, review this for basic usage of diffusion model and the triple-tenc cat
- `sd3_impls.py` - contains the wrapper around the MMDiT and the VAE
- `other_impls.py` - contains the CLIP model, the T5 model, and some utilities
- `mmdit.py` - contains the core of the MMDiT itself
- folder `models` with the following files (download separately):
    - `clip_g.safetensors` (openclip bigG, same as SDXL, can grab a public copy)
    - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL, can grab a public copy)
    - `t5xxl.safetensors` (google T5-v1.1-XXL, can grab a public copy)
    - `sd3_beta.safetensors` (internal, private)

### Legal

Built by Alex Goodwin for Stability AI and private partners under NDA, heavily based on internal ComfyUI and SGM codebases. Uses some upstream logic from HuggingFace, Google, PyTorch.

Do not redistribute.

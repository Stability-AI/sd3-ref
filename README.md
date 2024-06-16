# Stable Diffusion 3 Micro-Reference Implementation

Inference-only tiny reference implementation of SD3.

Contains code for the text encoders (OpenAI CLIP-L/14, OpenCLIP bigG, Google T5-XXL) (these models are all public), the VAE Decoder (similar to previous SD models, but 16-channels and no postquantconv step), and the core MM-DiT (entirely new).

Everything you need to inference SD3 excluding the weights files.

Note: this repo is an early reference lib meant to assist partner organizations in implementing SD3. For normal inference, use [Comfy](https://github.com/comfyanonymous/ComfyUI) or UIs based on it such as [Swarm](https://github.com/Stability-AI/StableSwarmUI).

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
python3 -s sd3_infer.py --width 1024 --height 1024 --shift 3 --model models/sd3_medium.safetensors --prompt "cute wallpaper art of a cat"
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
    - `sd3_medium.safetensors` (or whichever main MMDiT model file)

### Code Origin

The code included here originates from:
- Stability AI internal research code repository (MM-DiT)
- Public Stability AI repositories (eg VAE)
- Some unique code for this reference repo written by Alex Goodwin for Stability AI
- Some code from ComfyUI internal Stability impl of SD3 (for some code corrections and handlers)
- HuggingFace and upstream providers (for sections of CLIP/T5 code)

### Legal

MIT License

Copyright (c) 2024 Stability AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

#### Note

Some code in `other_impls` originates from HuggingFace and is subject to [the HuggingFace Transformers Apache2 License](https://github.com/huggingface/transformers/blob/main/LICENSE)

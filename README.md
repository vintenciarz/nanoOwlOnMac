<h1 align="center">nanoOwlOnMac</h1>

<p align="center">
  Run NanoOWL on macOS / Apple Silicon without TensorRT.
</p>

<p align="center">
  <a href="https://github.com/vintenciarz/nanoOwlOnMac">Repo</a> ·
  <a href="#quick-start-on-macos">Quick Start</a> ·
  <a href="#run-live-camera-detection">Live Camera</a> ·
  <a href="#optional-install-clip">CLIP Setup</a>
</p>

> This repository was created based on [NVIDIA-AI-IOT/nanoowl](https://github.com/NVIDIA-AI-IOT/nanoowl).

<p align="center">
  <img src="assets/jetson_person_2x.gif" width="70%" alt="Live detection demo" />
</p>

## Why This Fork Exists

The original NanoOWL project is designed around Jetson hardware, CUDA, and TensorRT.
This fork focuses on a different goal:

- make NanoOWL usable on Apple Silicon Macs,
- keep setup simple for local experimentation,
- support live prompt editing from the terminal,
- make hierarchical prompts practical in a webcam workflow.

If you want the Jetson-optimized version, use the original NVIDIA repository.

## What You Get

- object detection on still images,
- live camera detection on macOS,
- automatic `mps` / `cpu` device selection,
- runtime prompt updates without restarting the app,
- hierarchical prompts such as `[a face [a nose, an eye, a mouth]]`.

## About

This repo adapts [NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) to run on Apple Silicon Macs without TensorRT and without Jetson hardware.
Instead of the original `CUDA + TensorRT` path, it uses PyTorch on `mps` or `cpu`.

A hierarchical prompt like:

```text
[a face [a nose, an eye, a mouth]]
```

means:

1. detect a face,
2. then search for a nose, eye, and mouth inside the detected face region.

## Requirements

- macOS on Apple Silicon
- Python 3.11
- internet access the first time the model is downloaded from Hugging Face
- camera access in macOS if you want to use live mode

## Quick Start On macOS

```bash
git clone https://github.com/vintenciarz/nanoOwlOnMac
cd nanoOwlOnMac
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . --no-build-isolation
```

If you do not already have the basic Python dependencies installed, add them with:

```bash
.venv/bin/pip install transformers pillow matplotlib
```

Notes:

- this repo also relies on `torch`, `torchvision`, and `opencv-python`,
- in my setup those were already installed globally, which is why the virtualenv uses `--system-site-packages`.

## Optional: Install CLIP

`clip` is only needed for classification-style prompts that use parentheses, for example:

```text
(indoors, outdoors)
[a face (happy, sad)]
```

It is not required for plain detection prompts such as:

```text
[a face]
[a cell phone]
[a face [a nose, an eye, a mouth]]
```

If you want classification prompts, install CLIP in the environment:

```bash
.venv/bin/pip install git+https://github.com/openai/CLIP.git
```

If `ftfy`, `regex`, or `tqdm` are missing, install them too:

```bash
.venv/bin/pip install ftfy regex tqdm
```

## Run On A Still Image

Face detection example:

```bash
cd /path/to/nanoOwlOnMac
.venv/bin/python examples/face_detect_mac.py \
  --image assets/class.jpg \
  --output data/face_detect_out.jpg
```

Custom prompt example:

```bash
.venv/bin/python examples/face_detect_mac.py \
  --image /path/to/image.jpg \
  --output data/my_result.jpg \
  --prompt "[a cell phone]"
```

## Run Live Camera Detection

The simplest start:

```bash
cd /path/to/nanoOwlOnMac
.venv/bin/python examples/face_camera_mac.py --mirror
```

Once it is running, type a new prompt in the same terminal and press `Enter`, for example:

```text
[a face]
[a cell phone]
[a face [a nose, an eye, a mouth]]
```

To close the app:

```text
q
```

or press `Esc` in the OpenCV window.

## Camera Permission On macOS

If the camera does not open:

1. open `System Settings`,
2. go to `Privacy & Security`,
3. click `Camera`,
4. enable access for the app you are using to run the script:
   `Terminal`, `iTerm`, `Codex`, or a similar app.

If the app is not listed yet, run the script again and click `Allow` in the macOS permission popup.

## Example Prompts

Large objects:

```text
[a face]
[a cell phone]
[a laptop]
[a coffee mug]
```

Multiple objects:

```text
[a laptop, a keyboard]
```

Hierarchical prompts:

```text
[a face [a nose]]
[a face [a nose, an eye, a mouth]]
```

Important:

- `[a nose]` tries to find a nose in the entire frame and usually works poorly,
- `[a face [a nose]]` finds a face first and only then searches for a nose, which is much more reliable.

## How To Increase FPS

The biggest gains usually come from:

- using a smaller input resolution,
- setting `--skip-frames 2` or `--skip-frames 3`,
- using a simpler prompt,
- detecting fewer objects at once.

Examples:

```bash
.venv/bin/python examples/face_camera_mac.py \
  --mirror \
  --width 960 \
  --height 540 \
  --skip-frames 2 \
  --prompt "[a face]"
```

```bash
.venv/bin/python examples/face_camera_mac.py \
  --mirror \
  --width 640 \
  --height 360 \
  --skip-frames 3 \
  --prompt "[a face]"
```

## What Changed Compared To The Original Repo

Main changes compared to the original `NVIDIA-AI-IOT/nanoowl`:

- runs without TensorRT,
- automatic `mps` / `cpu` device selection,
- single-image detection script:
  `examples/face_detect_mac.py`,
- live macOS camera script:
  `examples/face_camera_mac.py`,
- runtime prompt updates from the terminal,
- hierarchical prompt support in live mode,
- drawing fixes for PIL / OpenCV image handling on macOS.

## Original Project

The original project is here:

- [NVIDIA-AI-IOT/nanoowl](https://github.com/NVIDIA-AI-IOT/nanoowl)

If you need the Jetson-optimized and TensorRT-based version, use the original NVIDIA repository.

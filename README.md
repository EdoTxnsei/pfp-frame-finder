# PFP Frame Finder

Find the best frames in a video to use as profile pictures, using a plain-text prompt like
*"dark, mysterious, red lighting"* or *"close-up of the singer"*. Works on any kind of video:
music videos, films, anime, gameplay, edits.

Analysis happens once (**index**), then you can **search** the video as many times as you like.

## How it works

1. **Index** (slow, once per video)
   - Scene detection (PySceneDetect), then the sharpest frame from each ~1s window
   - Skips black/blank frames and fade transitions; removes letterbox bars
   - Measures sharpness, brightness, contrast and color
   - Embeds every frame with CLIP (via `open_clip`)
2. **Search** (fast)
   - Embeds your prompt and ranks frames by: text relevance, an aesthetic proxy, technical quality, and explicit color/darkness filters parsed from the prompt
   - Removes near-duplicates with MMR (diversity) plus a minimum time gap
   - Writes a contact sheet and, optionally, full-resolution exports

## Install

Requires Python 3.10+. Install PyTorch **first** from [pytorch.org](https://pytorch.org)
(CPU, CUDA, or ROCm for AMD on Linux), then:

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The first run downloads the CLIP weights from the Hugging Face Hub (a few hundred MB, cached afterwards).

## Usage

Run from the folder that contains `pfpfinder/`.

```
# 1. Analyze a video once
python -m pfpfinder.cli index clip.mp4 idx

# 2. Search it
python -m pfpfinder.cli search idx "dark, mysterious, aura" -k 12

# 3. Search and export results
python -m pfpfinder.cli search idx "red lighting" -k 12 --export out
```

With `--export`, each search gets its own folder:

```
out/dark-mysterious-aura/
├── results.jpg     # contact sheet of the results
├── full/           # full-resolution frames
└── square/         # center-square crops
```

Useful flags: `--square` (square previews in the sheet), `--diversity 0.7` (1.0 = pure score order),
`--min-gap 1.5` (seconds between results), `--device cpu|cuda`.

## Roadmap

- Face/saliency-aware square cropping and crop-aware scoring
- Subtitle/watermark/logo penalty
- Quality floor to drop weak frames
- Face clustering and reference-image search
- Optional vision-language reranking
- API + web UI (FastAPI + Next.js) with gallery and cropper

## Notes

Only process videos you have the right to use. Videos are never uploaded anywhere;
everything runs locally.

"""Indexing: video -> candidate frames -> features + embeddings saved to disk.

Pipeline
  1. Scene detection (PySceneDetect)
  2. Letterbox/pillarbox detection (once per video)
  3. Per ~1s window inside each scene, look at a few frames and keep the
     sharpest valid one (avoids motion blur, black frames)
  4. Compute cheap features: sharpness, brightness, contrast, color fractions
  5. Embed every kept frame with CLIP/SigLIP and save everything
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, detect
from tqdm import tqdm

from .models import (
    DEFAULT_MODEL,
    DEFAULT_PRETRAINED,
    describe_device,
    embed_images,
    load_model,
    pick_device,
)

# OpenCV hue is 0-179
COLOR_RANGES = {
    "red": [(0, 8), (170, 179)],
    "orange": [(9, 20)],
    "yellow": [(21, 34)],
    "green": [(35, 84)],
    "cyan": [(85, 99)],
    "blue": [(100, 129)],
    "purple": [(130, 159)],
    "pink": [(160, 169)],
}


def detect_scenes(video_path, n_frames):
    try:
        scenes = detect(str(video_path), ContentDetector(threshold=27.0))
    except Exception as e:  # corrupt stream etc.
        print(f"  scene detection failed ({e}); treating video as one scene")
        scenes = []
    if not scenes:
        return [(0, n_frames)]
    return [(s.get_frames(), e.get_frames()) for s, e in scenes]


def detect_letterbox(cap, n=30, thresh=16):
    """Find black bars once per video by looking at 'ever bright' rows/cols."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    row_max = col_max = None
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r, c = g.mean(axis=1), g.mean(axis=0)
        row_max = r if row_max is None else np.maximum(row_max, r)
        col_max = c if col_max is None else np.maximum(col_max, c)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if row_max is None:
        return None
    rows, cols = np.where(row_max > thresh)[0], np.where(col_max > thresh)[0]
    if not len(rows) or not len(cols):
        return None
    h, w = len(row_max), len(col_max)
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
    if (y1 - y0) < 0.5 * h or (x1 - x0) < 0.5 * w:  # implausible, ignore
        return None
    return (x0, y0, x1, y1)


def build_windows(scenes, fps, window_sec=1.0, per_window=3, margin=2):
    """Return [(scene_id, [frame_idx, ...]), ...] in increasing frame order."""
    win = max(1, int(round(fps * window_sec)))
    windows = []
    for sid, (s, e) in enumerate(scenes):
        # skip the first/last frames of a shot: fades and transitions live there
        a, b = (s + margin, e - margin) if (e - s) > 2 * margin + 2 else (s, e)
        for w0 in range(a, b, win):
            w1 = min(b, w0 + win)
            if w1 <= w0:
                continue
            idxs = sorted(set(np.linspace(w0, w1 - 1, per_window).astype(int).tolist()))
            windows.append((sid, idxs))
    return windows


def color_fractions(bgr):
    small = cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colorful = (s > 90) & (v > 50)
    out = {}
    for name, ranges in COLOR_RANGES.items():
        m = np.zeros(h.shape, dtype=bool)
        for lo, hi in ranges:
            m |= (h >= lo) & (h <= hi)
        out[name] = float((m & colorful).mean())
    return out


def frame_stats(crop):
    scale = 640 / max(crop.shape[:2])
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "bright": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def index_video(
    video_path,
    out_dir,
    model_name=DEFAULT_MODEL,
    pretrained=DEFAULT_PRETRAINED,
    device=None,
    window_sec=1.0,
    per_window=3,
    thumb_size=512,
):
    out = Path(out_dir)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H}, {fps:.2f} fps, {n_frames} frames (~{n_frames / fps:.0f}s)")

    print("Detecting scenes...")
    scenes = detect_scenes(video_path, n_frames)
    print(f"  {len(scenes)} scenes")
    box = detect_letterbox(cap) or (0, 0, W, H)
    if box != (0, 0, W, H):
        print(f"  letterbox detected, content box = {box}")
    x0, y0, x1, y1 = box

    windows = build_windows(scenes, fps, window_sec, per_window)
    print(f"Sampling {len(windows)} windows...")

    records, pos = [], 0
    for sid, idxs in tqdm(windows):
        best = None
        for idx in idxs:
            if idx < pos:
                continue
            while pos < idx:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            pos += 1
            if not ok:
                continue
            crop = frame[y0:y1, x0:x1]
            st = frame_stats(crop)
            if st["bright"] < 8 or st["contrast"] < 5:  # black / blank / mid-fade
                continue
            if best is None or st["sharp"] > best[1]["sharp"]:
                best = (idx, st, crop)
        if best is None:
            continue
        idx, st, crop = best
        k = len(records)
        scale = thumb_size / max(crop.shape[:2])
        thumb = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
        name = f"{k:05d}.jpg"
        cv2.imwrite(str(out / "thumbs" / name), thumb, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append(
            {"id": k, "frame": int(idx), "t": idx / fps, "scene": sid, "thumb": name,
             "colors": color_fractions(crop), **st}
        )
    cap.release()
    print(f"Kept {len(records)} candidate frames")

    device = pick_device(device)
    print(f"Embedding on {describe_device(device)} with {model_name}/{pretrained}...")
    model, preprocess, _ = load_model(model_name, pretrained, device)
    paths = [str(out / "thumbs" / r["thumb"]) for r in records]
    emb = embed_images(model, preprocess, paths, device)
    np.save(out / "embeddings.npy", emb)

    meta = {
        "video": str(video_path), "fps": fps, "width": W, "height": H, "box": list(box),
        "model": model_name, "pretrained": pretrained, "frames": records,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"Index saved to {out}")

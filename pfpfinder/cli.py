"""Command line interface.

  python -m pfpfinder.cli index  my_video.mp4 idx/
  python -m pfpfinder.cli search idx/ "dark, mysterious, red lighting" -k 12
  python -m pfpfinder.cli search idx/ "close-up of the singer" --export out/
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from .indexer import index_video
from .models import DEFAULT_MODEL, DEFAULT_PRETRAINED, describe_device, load_model, pick_device
from .search import VideoIndex, search


def fmt_time(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def contact_sheet(results, path, cols=4, cell_w=400, square=False):
    tiles = []
    for r in results:
        img = cv2.imread(r["thumb"])
        if square:  # preview what the center-square PFP crop will look like
            h, w = img.shape[:2]
            side = min(h, w)
            y, x = (h - side) // 2, (w - side) // 2
            img = img[y : y + side, x : x + side]
        h, w = img.shape[:2]
        img = cv2.resize(img, (cell_w, int(h * cell_w / w)), interpolation=cv2.INTER_AREA)
        label = f"#{r['rank']} {fmt_time(r['t'])} {r['score']:.2f}"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(img)
    cell_h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        canvas = np.zeros((cell_h, cell_w, 3), np.uint8)
        canvas[: t.shape[0], : t.shape[1]] = t
        padded.append(canvas)
    while len(padded) % cols:
        padded.append(np.zeros((cell_h, cell_w, 3), np.uint8))
    rows = [np.hstack(padded[i : i + cols]) for i in range(0, len(padded), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def make_run_dir(base, query):
    """<base>/<query-slug>/ ; adds -2, -3... if that folder already exists."""
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40] or "search"
    run = Path(base) / slug
    n = 2
    while run.exists():
        run = Path(base) / f"{slug}-{n}"
        n += 1
    run.mkdir(parents=True)
    return run


def export_frames(index, results, run_dir):
    """Re-extract each frame from the source at full resolution.

    <run_dir>/full/    whole frame (black bars removed)
    <run_dir>/square/  center-square crop
    """
    run = Path(run_dir)
    (run / "full").mkdir(parents=True, exist_ok=True)
    (run / "square").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(index.meta["video"])
    x0, y0, x1, y1 = index.meta["box"]
    for r in results:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[y0:y1, x0:x1]
        h, w = crop.shape[:2]
        s = min(h, w)
        cx, cy = (w - s) // 2, (h - s) // 2  # TODO: face/saliency-aware crop
        sq = crop[cy : cy + s, cx : cx + s]
        name = f"pfp_{r['rank']:02d}_{fmt_time(r['t']).replace(':', '-')}.png"
        cv2.imwrite(str(run / "full" / name), crop)
        cv2.imwrite(str(run / "square" / name), sq)
    cap.release()


def main():
    p = argparse.ArgumentParser(prog="pfpfinder")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="analyze a video and build an index")
    pi.add_argument("video")
    pi.add_argument("out")
    pi.add_argument("--model", default=DEFAULT_MODEL)
    pi.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    pi.add_argument("--device", default=None, help="cpu or cuda (cuda also means ROCm)")
    pi.add_argument("--window", type=float, default=1.0, help="seconds per sampling window")

    ps = sub.add_parser("search", help="search an index with a text prompt")
    ps.add_argument("index")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=12)
    ps.add_argument("--diversity", type=float, default=0.7, help="MMR lambda: 1.0 = pure score, lower = more diverse")
    ps.add_argument("--min-gap", type=float, default=1.5, help="min seconds between results")
    ps.add_argument("--device", default=None)
    ps.add_argument("--sheet", default=None, help="contact sheet path (default: <index>/results.jpg)")
    ps.add_argument("--square", action="store_true", help="show square-crop previews in the contact sheet")
    ps.add_argument("--export", default=None, help="base folder; each search gets its own subfolder with results.jpg, full/ and square/")

    a = p.parse_args()

    if a.cmd == "index":
        index_video(a.video, a.out, a.model, a.pretrained, a.device, window_sec=a.window)
        return

    idx = VideoIndex(a.index)
    device = pick_device(a.device)
    print(f"Using {describe_device(device)}")
    model, _, tok = load_model(idx.meta["model"], idx.meta["pretrained"], device)
    results, parsed = search(idx, model, tok, device, a.query, a.k, a.diversity, a.min_gap)
    if parsed["colors"] or parsed["dark"]:
        print(f"Detected filters: colors={parsed['colors']} dark={parsed['dark']}")
    for r in results:
        parts = " ".join(f"{k}={v:.2f}" for k, v in r["parts"].items())
        print(f"#{r['rank']:>2}  {fmt_time(r['t'])}  score={r['score']:.3f}  [{parts}]")
    if a.export:
        run_dir = make_run_dir(a.export, a.query)
        sheet = a.sheet or str(run_dir / "results.jpg")
        contact_sheet(results, sheet, square=a.square)
        export_frames(idx, results, run_dir)
        print(f"Saved to {run_dir}  (results.jpg, full/, square/)")
    else:
        sheet = a.sheet or str(Path(a.index) / "results.jpg")
        contact_sheet(results, sheet, square=a.square)
        print(f"Contact sheet: {sheet}")


if __name__ == "__main__":
    main()
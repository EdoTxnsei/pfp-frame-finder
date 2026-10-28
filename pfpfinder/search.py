"""Search an index: CLIP text similarity + aesthetic proxy + technical quality
+ explicit color/darkness filters, then MMR for diversity."""
import json
import re
from pathlib import Path

import numpy as np

from .models import embed_texts

COLOR_ALIASES = {
    "red": "red", "crimson": "red", "scarlet": "red",
    "orange": "orange", "yellow": "yellow", "golden": "yellow", "gold": "yellow",
    "green": "green", "cyan": "cyan", "teal": "cyan", "turquoise": "cyan",
    "blue": "blue", "purple": "purple", "violet": "purple",
    "pink": "pink", "magenta": "pink",
}
DARK_WORDS = {"dark", "moody", "noir", "dim", "shadowy", "black"}

# Zero-shot "aesthetic" proxy. Swap for the LAION aesthetic predictor later.
AES_POS = [
    "a beautiful, high quality, professionally composed photo",
    "a stunning cinematic still with great lighting",
]
AES_NEG = [
    "a blurry, low quality, ugly photo",
    "a badly composed, boring, washed out image",
    "a screenshot with subtitles, text overlays and logos",
]


class VideoIndex:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.emb = np.load(self.dir / "embeddings.npy")
        self.frames = self.meta["frames"]
        self.t = np.array([f["t"] for f in self.frames])
        self.sharp = np.array([f["sharp"] for f in self.frames])
        self.bright = np.array([f["bright"] for f in self.frames])


def pct(x):
    """Percentile rank in [0, 1] (scores become comparable across signals)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.ones(len(x))
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def parse_query(query):
    words = re.findall(r"[a-z]+", query.lower())
    colors = sorted({COLOR_ALIASES[w] for w in words if w in COLOR_ALIASES})
    dark = any(w in DARK_WORDS for w in words)
    return colors, dark


def mmr_select(score, emb, times, k, lam=0.7, min_gap=1.5):
    cand = list(np.argsort(-score)[: max(200, k * 10)])
    chosen = []
    while len(chosen) < k and cand:
        best, best_v = None, -1e9
        for i in cand:
            if any(abs(times[i] - times[j]) < min_gap for j in chosen):
                continue
            sim = float((emb[chosen] @ emb[i]).max()) if chosen else 0.0
            diversity_penalty = float(np.clip((sim - 0.6) / 0.4, 0, 1))
            v = lam * score[i] - (1 - lam) * diversity_penalty
            if v > best_v:
                best, best_v = i, v
        if best is None:
            break
        chosen.append(best)
        cand.remove(best)
    return chosen


def search(index, model, tokenizer, device, query, k=12, lam=0.7, min_gap=1.5):
    q = embed_texts(model, tokenizer, [query], device)[0]
    rel = pct(index.emb @ q)

    pos = embed_texts(model, tokenizer, AES_POS, device)
    neg = embed_texts(model, tokenizer, AES_NEG, device)
    aes = pct((index.emb @ pos.T).mean(1) - (index.emb @ neg.T).mean(1))

    # technical quality: sharpness rank, with a penalty for crushed/blown exposure
    exposure_ok = np.clip(index.bright / 25, 0, 1) * np.clip((250 - index.bright) / 25, 0, 1)
    tech = pct(np.log1p(index.sharp)) * exposure_ok

    parts = {"rel": rel, "aes": aes, "tech": tech}
    w = {"rel": 0.60, "aes": 0.25, "tech": 0.15}

    colors, dark = parse_query(query)
    if colors:
        frac = np.array([np.mean([f["colors"][c] for c in colors]) for f in index.frames])
        parts["color"] = np.clip(frac / 0.20, 0, 1)
        w.update(rel=0.45, aes=0.20, tech=0.10, color=0.25)
    if dark:
        parts["dark"] = pct(-index.bright)
        w["dark"] = 0.15

    score = sum(w[n] * parts[n] for n in w) / sum(w.values())
    chosen = mmr_select(score, index.emb, index.t, k, lam, min_gap)

    results = []
    for rank, i in enumerate(chosen, 1):
        f = index.frames[i]
        results.append({
            "rank": rank, "id": f["id"], "frame": f["frame"], "t": f["t"],
            "score": float(score[i]),
            "parts": {n: float(parts[n][i]) for n in parts},
            "thumb": str(index.dir / "thumbs" / f["thumb"]),
        })
    return results, {"colors": colors, "dark": dark}

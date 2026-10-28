"""Model helpers.

Works on CPU, NVIDIA (CUDA) and AMD (ROCm). PyTorch's ROCm builds expose the
GPU through the "cuda" device name, so no special-casing is needed.
"""
import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


def pick_device(force=None):
    if force:
        return force
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device):
    if device == "cuda":
        backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
        return f"{torch.cuda.get_device_name(0)} ({backend})"
    return "CPU"


def load_model(name=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, device="cpu"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(name)
    return model, preprocess, tokenizer


def _normalize(f):
    f = f.float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()


@torch.no_grad()
def embed_images(model, preprocess, paths, device, batch_size=32):
    out = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]]
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            f = model.encode_image(batch)
        out.append(_normalize(f))
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
        f = model.encode_text(tokens)
    return _normalize(f)

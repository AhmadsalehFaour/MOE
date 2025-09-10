from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

try:
    import cv2  # optional
except Exception:
    cv2 = None

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IMAGENET_LABELS = None

def load_imagenet_labels():
    global IMAGENET_LABELS
    if IMAGENET_LABELS is not None:
        return IMAGENET_LABELS
    try:
        from urllib.request import urlopen
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        labels = urlopen(url).read().decode("utf-8").strip().split("\n")
    except Exception:
        labels = [f"class_{i}" for i in range(1000)]
    IMAGENET_LABELS = labels
    return labels

COLOR_NAMES = [
    (0.00, "red"), (0.08, "orange"), (0.16, "yellow"), (0.24, "lime"), (0.33, "green"),
    (0.41, "teal"), (0.50, "cyan"), (0.58, "sky"), (0.66, "blue"), (0.75, "indigo"),
    (0.83, "violet"), (0.91, "magenta"), (0.99, "red")
]

def hue_to_name(h: float) -> str:
    best = min(COLOR_NAMES, key=lambda kv: abs(kv[0]-h))
    return best[1]

def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    mx = arr.max(-1)
    mn = arr.min(-1)
    diff = mx - mn + 1e-12
    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = ((g - b)[mask] / diff[mask]) % 6
    mask = mx == g
    h[mask] = ((b - r)[mask] / diff[mask]) + 2
    mask = mx == b
    h[mask] = ((r - g)[mask] / diff[mask]) + 4
    h = (h / 6.0) % 1.0
    s = diff / (mx + 1e-12)
    v = mx
    return np.stack([h, s, v], axis=-1)

def dominant_colors(img: Image.Image, k: int = 3):
    small = img.convert("RGB").resize((96, 96))
    arr = np.asarray(small) / 255.0
    hsv = rgb_to_hsv(arr)
    h = hsv[..., 0].reshape(-1)
    bins = np.linspace(0, 1, 13)
    idx = np.digitize(h, bins)
    counts = np.bincount(idx, minlength=len(bins)+1)
    top = counts.argsort()[::-1][:k]
    names = [hue_to_name((b-0.5)/12) for b in top]
    return list(dict.fromkeys(names))

def image_complexity(img: Image.Image) -> float:
    if cv2 is None:
        g = np.asarray(img.convert("L").resize((128,128)))
        hist, _ = np.histogram(g, bins=64, range=(0,255), density=True)
        ent = -np.sum(hist * (np.log(hist + 1e-12)))
        return float(min(1.0, ent / 4.5))
    arr = np.asarray(img.convert("L"))
    edges = cv2.Canny(arr, 80, 160)
    density = edges.mean() / 255.0
    return float(np.clip(density * 3.0, 0.0, 1.0))

@dataclass
class VisionOutput:
    topk_labels: List[Tuple[str, float]]
    entropy: float
    p_max: float
    colors: List[str]
    complexity: float

class VisionExpert:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        self.model.eval()
        self.labels = load_imagenet_labels()

    @torch.inference_mode()
    def __call__(self, img_pil: Image.Image, topk: int = 5) -> VisionOutput:
        x = IMAGENET_TRANSFORM(img_pil).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        p_max = float(probs.max().item())
        entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
        topk_vals, topk_idx = torch.topk(probs, k=topk)
        pairs = [(self.labels[int(i)], float(v)) for v, i in zip(topk_vals.tolist(), topk_idx.tolist())]
        colors = dominant_colors(img_pil)
        complexity = image_complexity(img_pil)
        return VisionOutput(pairs, entropy, p_max, colors, complexity)

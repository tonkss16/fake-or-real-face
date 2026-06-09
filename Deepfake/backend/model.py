import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
from typing import Dict, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self, model: nn.Module, idx_to_class: Dict[int, str], transform):
        self.model = model
        self.idx_to_class = idx_to_class
        self.transform = transform


def load_model(pth_path: str) -> Predictor:
    ckpt = torch.load(pth_path, map_location=device)

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()

    idx_to_class = {0: "fake", 1: "real"}
    if isinstance(ckpt, dict) and "class_to_idx" in ckpt:
        idx_to_class = {int(v): str(k) for k, v in ckpt["class_to_idx"].items()}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return Predictor(model, idx_to_class, transform)


def predict_pil_image(
    predictor: Predictor,
    pil_img: Image.Image
) -> Tuple[str, float, Dict[str, float], list, Dict[int, str]]:
    x = predictor.transform(pil_img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(predictor.model(x), dim=1)[0]

    p0 = float(probs[0])
    p1 = float(probs[1])

    scores = {
        predictor.idx_to_class[0]: p0,
        predictor.idx_to_class[1]: p1,
    }

    pred_idx = int(torch.argmax(probs))
    label = predictor.idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    return label, confidence, scores, [p0, p1], predictor.idx_to_class

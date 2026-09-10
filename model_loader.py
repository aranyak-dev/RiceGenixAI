import os
import json

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

DEFAULT_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight",
    "Tungro",
]

if TORCH_AVAILABLE:
    def build_model(num_classes=9):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


def load_model():
    if not TORCH_AVAILABLE:
        return None

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "disease_model.pth")
        classes_path = os.path.join(base_dir, "model", "disease_classes.json")

        class_names = DEFAULT_CLASSES

        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, list) and len(loaded) >= 2:
                class_names = loaded

        checkpoint = torch.load(model_path, map_location="cpu")

        model = build_model(num_classes=len(class_names))

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])

            if isinstance(checkpoint.get("class_names"), list):
                class_names = checkpoint["class_names"]
        else:
            model.load_state_dict(checkpoint)

        model.class_names = class_names
        model.image_size = 192
        model.eval()

        return model

    except Exception as exc:
        print(f"RiceGenixAI disease model load error: {exc}")
        return None


def _preprocess(image, image_size=192):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    return transform(image.convert("RGB")).unsqueeze(0)


def predict_with_confidence(model, image):
    if not TORCH_AVAILABLE or model is None:
        return "AI Model Not Available", 0.0

    class_names = getattr(model, "class_names", DEFAULT_CLASSES)
    image_size = getattr(model, "image_size", 192)

    tensor = _preprocess(image, image_size)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, dim=1)

    index = pred.item()
    confidence_percent = float(confidence.item() * 100)

    if 0 <= index < len(class_names):
        return class_names[index], confidence_percent

    return "Unknown", confidence_percent


def predict(model, image):
    disease, _ = predict_with_confidence(model, image)
    return disease
import os
import json

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

DEFAULT_CLASSES = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy", "Leaf Blast",
    "Leaf Scald", "Narrow Brown Leaf Spot", "Rice Hispa", "Sheath Blight"
]

if TORCH_AVAILABLE:
    class RiceDiseaseModel(nn.Module):
        """ResNet18 classifier used by the accuracy-focused model."""
        def __init__(self, num_classes=8):
            super().__init__()
            self.backbone = models.resnet18(weights=None)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        def forward(self, x):
            return self.backbone(x)


def _load_classes(model_dir, output_count):
    metadata = os.path.join(model_dir, "disease_classes.json")
    if os.path.exists(metadata):
        try:
            with open(metadata, "r", encoding="utf-8") as f:
                classes = json.load(f)
            if isinstance(classes, list) and len(classes) == output_count:
                return classes
        except Exception:
            pass
    classes = DEFAULT_CLASSES.copy()
    if output_count == 9:
        classes.append("Tungro")
    return classes[:output_count]


def load_model():
    if not TORCH_AVAILABLE:
        return None
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        model_path = os.path.join(model_dir, "disease_model.pth")
        checkpoint = torch.load(model_path, map_location="cpu")

        # New checkpoint: state_dict plus metadata.
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        fc_weight = state_dict.get("backbone.fc.weight")
        if fc_weight is None:
            # Legacy 8-class CNN checkpoint: retain backward compatibility.
            class LegacyRiceDiseaseModel(nn.Module):
                def __init__(self, num_classes=8):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2)
                    )
                    self.fc = nn.Sequential(
                        nn.Flatten(), nn.Linear(64*26*26,128), nn.ReLU(),
                        nn.Linear(128,num_classes)
                    )
                def forward(self, x):
                    return self.fc(self.conv(x))
            output_count = state_dict["fc.3.weight"].shape[0]
            model = LegacyRiceDiseaseModel(output_count)
            model.load_state_dict(state_dict)
        else:
            output_count = fc_weight.shape[0]
            model = RiceDiseaseModel(output_count)
            model.load_state_dict(state_dict)

        model.eval()
        model._rice_classes = _load_classes(model_dir, output_count)
        return model
    except Exception:
        return None


def _prepare(image):
    # ImageNet normalization is required by the pretrained ResNet18 pipeline.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(model, image):
    if not TORCH_AVAILABLE or model is None:
        return "AI Model Not Available"
    try:
        with torch.no_grad():
            output = model(_prepare(image))
            pred = torch.argmax(output, dim=1).item()
        classes = getattr(model, "_rice_classes", DEFAULT_CLASSES)
        return classes[pred] if 0 <= pred < len(classes) else "AI Model Not Available"
    except Exception:
        return "AI Model Not Available"


def predict_with_confidence(model, image):
    if not TORCH_AVAILABLE or model is None:
        return "AI Model Not Available", 0.0
    try:
        with torch.no_grad():
            logits = model(_prepare(image))
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())
            confidence = float(probs[pred].item()) * 100.0
        classes = getattr(model, "_rice_classes", DEFAULT_CLASSES)
        label = classes[pred] if pred < len(classes) else "AI Model Not Available"
        return label, confidence
    except Exception:
        return "AI Model Not Available", 0.0

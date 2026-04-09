# ---------------- SAFE IMPORT ----------------
import os

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------------- MODEL DEFINITION ----------------
if TORCH_AVAILABLE:

    class RiceDiseaseModel(nn.Module):
        def __init__(self, num_classes=8):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(3,16,3),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16,32,3),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32,64,3),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*26*26,128),
                nn.ReLU(),
                nn.Linear(128,num_classes)
            )

        def forward(self,x):
            x = self.conv(x)
            x = self.fc(x)
            return x

# ---------------- LOAD MODEL ----------------
def load_model():
    if not TORCH_AVAILABLE:
        return None

    try:
        model_path = os.path.join(os.path.dirname(__file__), "model", "disease_model.pth")
        model = RiceDiseaseModel(num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except Exception:
        return None

# ---------------- PREDICT ----------------
def predict(model, image):
    if not TORCH_AVAILABLE or model is None:
        return "AI Model Not Available"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output,1).item()

    classes = [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy",
        "Leaf Blast",
        "Leaf Scald",
        "Narrow Brown Leaf Spot",
        "Rice Hispa",
        "Sheath Blight"
    ]

    return classes[pred]

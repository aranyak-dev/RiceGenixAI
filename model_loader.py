import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------- MODEL CLASS ----------------
class RiceDiseaseModel(nn.Module):
    def __init__(self, num_classes=8):
        super(RiceDiseaseModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*26*26, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ---------------- LOAD MODEL ----------------
def load_model():
    model = RiceDiseaseModel(num_classes=8)

    model.load_state_dict(torch.load("model/disease_model.pth", map_location=torch.device('cpu')))
    model.eval()

    return model

# ---------------- CLASS NAMES ----------------
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]

# ---------------- PREDICT FUNCTION ----------------
def predict(image, model):
    if model is None:
        return "Model Not Loaded"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]
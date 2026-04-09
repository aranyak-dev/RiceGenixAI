import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# ---------------- DATASET ----------------
train_path = "dataset/train"

if not os.path.exists(train_path):
    raise Exception("Training folder not found!")

train_dataset = datasets.ImageFolder(train_path, transform=transform)

print("Classes found:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

if len(train_dataset.classes) < 2:
    raise Exception("ERROR: You have less than 2 classes. Fix dataset folder!")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ---------------- MODEL ----------------
class RiceDiseaseModel(nn.Module):
    def __init__(self, num_classes):
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

model = RiceDiseaseModel(num_classes=len(train_dataset.classes)).to(device)

# ---------------- TRAINING SETUP ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    running_loss = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} COMPLETE | Total Loss: {running_loss:.4f}")
    print("-"*50)

# ---------------- SAVE MODEL ----------------
os.makedirs("model", exist_ok=True)

torch.save(model.state_dict(), "model/disease_model.pth")

print("Model saved successfully!")
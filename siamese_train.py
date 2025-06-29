"""
Train a Siamese Neural Network to detect ATM tampering
------------------------------------------------------

This script builds and trains a Siamese Network that learns to distinguish 
between clean and tampered ATM images by comparing image pairs. It uses 
ResNet18 as the base encoder and trains on image pairs labeled as either 
similar (both clean) or different (clean vs tampered).

Outputs:
    - Trained model weights saved as 'siamese_atm_model.pth'

Dependencies:
    - torchvision
    - PIL
    - torch
    - tqdm
"""

import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# === Paths to image folders ===
clean_dir = "data/clean_atms"
tampered_dir = "data/tampered_atms"

# === Hyperparameters ===
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Preprocessing transformation for input images ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# === Siamese Dataset Loader ===
class SiameseATMDataset(Dataset):
    """
    Dataset for generating pairs of ATM images:
    - Positive pair: two clean ATM images (label 0)
    - Negative pair: clean ATM vs tampered ATM (label 1)
    """
    def __init__(self, clean_dir, tampered_dir, transform):
        self.clean_images = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)]
        self.tampered_images = [os.path.join(tampered_dir, f) for f in os.listdir(tampered_dir)]
        self.transform = transform
        self.pairs = []

        # Generate positive pairs: (clean, clean)
        for i in range(len(self.clean_images)):
            img1 = self.clean_images[i]
            img2 = random.choice(self.clean_images)
            self.pairs.append((img1, img2, 0))  # Label 0 = similar

        # Generate negative pairs: (clean, tampered)
        for t in self.tampered_images:
            base_name = os.path.basename(t).split("_")[0]  # Match by prefix
            match = [f for f in self.clean_images if base_name in os.path.basename(f)]
            if match:
                self.pairs.append((match[0], t, 1))  # Label 1 = different

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = self.transform(Image.open(img1_path).convert("RGB"))
        img2 = self.transform(Image.open(img2_path).convert("RGB"))
        return img1, img2, torch.tensor([label], dtype=torch.float32)

# === Siamese Network Definition ===
class SiameseNetwork(nn.Module):
    """
    Siamese neural network using ResNet18 backbone to learn image similarity.
    Outputs a score between 0 (similar) and 1 (different).
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base_cnn = models.resnet18(pretrained=True)
        base_cnn.fc = nn.Identity()  # Remove final classification layer
        self.embedding = base_cnn

        # Comparison head: converts feature distance to similarity score
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        f1 = self.embedding(x1)           # Feature for image 1
        f2 = self.embedding(x2)           # Feature for image 2
        dist = torch.abs(f1 - f2)         # Absolute difference
        out = self.fc(dist)               # Final similarity score
        return out

# === Training Loop ===
def train_model():
    """
    Loads the dataset, initializes the model and trains it on image pairs.
    """
    dataset = SiameseATMDataset(clean_dir, tampered_dir, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SiameseNetwork().to(DEVICE)
    criterion = nn.BCELoss()  # Binary classification: tampered or not
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for img1, img2, label in tqdm(dataloader):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            output = model(img1, img2)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), "siamese_atm_model.pth")
    print("✅ Model saved to siamese_atm_model.pth")

# === Main Entry Point ===
if __name__ == "__main__":
    train_model()
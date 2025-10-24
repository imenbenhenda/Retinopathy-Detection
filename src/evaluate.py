# =============================================================================
# 1. Imports
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 2. Custom Dataset Class
# =============================================================================
class RetinoDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading retinopathy images."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        # Binary: 0 = healthy, 1 = diseased
        self.data["label"] = self.data["diagnosis"].apply(lambda x: 0 if x == 0 else 1)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================================================
# 3. Validation Transforms
# =============================================================================
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# =============================================================================
# 4. Evaluation Function
# =============================================================================
def evaluate(model, data_loader, criterion, device):
    """Evaluates the model and prints detailed metrics."""
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_loss /= len(data_loader)
    val_acc = correct / total
    print(f"\n--- Evaluation Results ---")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} ({correct}/{total})")

    # Classification report
    class_names = ['Healthy (0)', 'Diseased (1)']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Plot confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    # Save the matrix (important for GitHub)
    os.makedirs("../models", exist_ok=True)
    plt.savefig("../models/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to '../models/confusion_matrix.png'")
    plt.show()

# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    
    # --- Configuration ---
    NUM_WORKERS = 4 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BEST_MODEL_PATH = "../models/resnet50_best_checkpoint.pth"
    print(f"Using device: {DEVICE}")

    # --- Load Dataset ---
    dataset = RetinoDataset("../data/train.csv", "../data/train_images", transform=val_transform)
    
  
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    
    _, val_dataset = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    # --- Dataloader ---
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    # --- Load Model Architecture ---
    # Load the empty architecture 
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(DEVICE)

    # --- Load Best Trained Weights ---
    print(f"Loading model weights from: {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()

    # --- Evaluate ---
    print("Starting evaluation...")
    evaluate(model, val_loader, criterion, DEVICE)
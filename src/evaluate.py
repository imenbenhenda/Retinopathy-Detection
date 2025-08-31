import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# === Dataset personnalisé (comme dans train.py) ===
class RetinoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
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


# === Transform pour validation/test ===
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# === Fonction d'évaluation ===
def evaluate(model, data_loader, criterion, device):
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
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Plot matrice de confusion
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # === Dataset ===
    dataset = RetinoDataset("../data/train.csv", "../data/train_images", transform=val_transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé:", device)

    # === Charger modèle ResNet50 ===
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    # === Charger les poids du meilleur modèle de fine-tuning ===
    model.load_state_dict(torch.load("../models/resnet50_best_finetune.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()

    # === Évaluer ===
    evaluate(model, val_loader, criterion, device)

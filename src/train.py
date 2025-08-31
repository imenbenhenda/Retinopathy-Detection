import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
import pickle

# === Dataset personnalisé ===
class RetinoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        # Binaire : 0 = sain, 1 = malade
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


# === Transforms pour l'entraînement et validation ===
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def train_one_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, phase_name):
    """Entraîne le modèle pour une phase donnée"""
    best_val_loss = float('inf')
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                print(f"[{phase_name}] Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"[{phase_name}] Epoch [{epoch+1}/{num_epochs}] "
              f"-> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"../models/resnet50_best_{phase_name}.pth")
            print(f"Meilleur modèle sauvegardé ({phase_name}) !")

    return train_losses, val_losses, val_accuracies


if __name__ == "__main__":
    # === Charger dataset ===
    dataset = RetinoDataset("../data/train.csv", "../data/train_images", transform=train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilisé:", device)

    # === Modèle ResNet50 ===
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    # === Phase 1 : Freeze backbone, entraîne seulement la tête ===
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("\n===== Phase 1 : Entraînement de la tête =====")
    t_losses1, v_losses1, v_accs1 = train_one_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=3, phase_name="head"
    )

    # === Phase 2 : Fine-tuning, on dégèle certaines couches ===
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:  # dernier bloc + fc
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("\n===== Phase 2 : Fine-tuning =====")
    t_losses2, v_losses2, v_accs2 = train_one_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=7, phase_name="finetune"
    )

    # === Sauvegarde de l’historique global ===
    history = {
        "train_loss": t_losses1 + t_losses2,
        "val_loss": v_losses1 + v_losses2,
        "val_acc": v_accs1 + v_accs2
    }
    os.makedirs("../models", exist_ok=True)
    with open("../models/history.pkl", "wb") as f:
        pickle.dump(history, f)

    print("\nEntraînement terminé avec succès ! Historique sauvegardé.")

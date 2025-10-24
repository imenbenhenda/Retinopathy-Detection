import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random

# =============================================================================
# 1. Reproducibility
# =============================================================================

def set_seed(seed):
    """Sets the seed for reproducibility of results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Warning: may slow down training
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# 2. Early Stopping Class
# =============================================================================

class EarlyStopping:
    """Stops training if validation loss doesn't improve after
    a given number of epochs (patience)."""
    def __init__(self, patience=3, delta=0, save_path="best_model.pth", verbose=True):
        self.patience = patience
        self.delta = delta # Minimum change to be considered an improvement
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # Performance did not improve
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Performance improved
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

# =============================================================================
# 3. Custom Dataset Class
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
# 4. Data Transforms
# =============================================================================

# Transforms for training data (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Transforms for validation data (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# =============================================================================
# 5. Training and Validation Function
# =============================================================================

def train_one_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, phase_name, early_stopper):
    """Trains the model for one phase with early stopping."""
    
    # We do NOT reset best_val_loss here,
    # as early_stopper manages it globally across phases.
    
    train_losses, val_losses, val_accuracies = [], [], []
    train_accuracies = []

    for epoch in range(num_epochs):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                print(f"[{phase_name}] Epoch [{epoch+1}/{num_epochs}], "
                      f"Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        # --- Validation Loop ---
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
              f"-> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        train_accuracies.append(train_acc)

        scheduler.step()

        # Call Early Stopping
        # This will save the model if val_loss improves
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"[{phase_name}] Early Stopping at epoch {epoch+1}")
            break # Exit epoch loop

    return train_losses, val_losses, val_accuracies, train_accuracies

# =============================================================================
# 6. Plotting Function
# =============================================================================

def plot_training_history(history, phase1_len, phase2_len):
    """Generates and saves the loss and accuracy plots."""
    
    total_epochs = len(history['train_loss'])
    if total_epochs == 0:
        print("History is empty, cannot generate plots.")
        return
        
    epochs = range(1, total_epochs + 1)
    phase_sep = phase1_len
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-', marker='.', markersize=4, label='Train Loss', alpha=0.7)
    ax1.plot(epochs, history['val_loss'], 'r-', marker='.', markersize=4, label='Val Loss', alpha=0.7)
    if phase_sep > 0 and phase_sep < total_epochs:
        ax1.axvline(x=phase_sep + 0.5, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy Plot
    ax2.plot(epochs, history['train_acc'], 'b-', marker='.', markersize=4, label='Train Accuracy', alpha=0.7)
    ax2.plot(epochs, history['val_acc'], 'r-', marker='.', markersize=4, label='Val Accuracy', alpha=0.7)
    if phase_sep > 0 and phase_sep < total_epochs:
        ax2.axvline(x=phase_sep + 0.5, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Annotations
    ylim_loss = ax1.get_ylim()
    ylim_acc = ax2.get_ylim()
    
    ax1.text(phase_sep/2, ylim_loss[1]*0.9, 'Phase 1\n(Head only)',
             ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    if phase2_len > 0:
      ax1.text(phase_sep + phase2_len/2, ylim_loss[1]*0.9, 'Phase 2\n(Fine-tuning)',
               ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax2.text(phase_sep/2, ylim_acc[0] + (ylim_acc[1]-ylim_acc[0])*0.1, 'Phase 1\n(Head only)',
             ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    if phase2_len > 0:
      ax2.text(phase_sep + phase2_len/2, ylim_acc[0] + (ylim_acc[1]-ylim_acc[0])*0.1, 'Phase 2\n(Fine-tuning)',
               ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    
    # Saving plots
    os.makedirs("../models", exist_ok=True)
    plt.savefig("../models/training_history.png", dpi=300, bbox_inches='tight')
    plt.savefig("../models/training_history.pdf", bbox_inches='tight')
    plt.show()
    
    print("Plots saved to '../models/training_history.png' and '../models/training_history.pdf'")

# =============================================================================
# 7. Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # --- Configuration ---
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 # Use 0 if you get errors, but 4 (or more) is faster
    BEST_MODEL_PATH = "../models/resnet50_best_checkpoint.pth"
    print(f"Using device: {DEVICE}")

    # --- Load Dataset ---
    dataset = RetinoDataset("../data/train.csv", "../data/train_images", transform=train_transform)
    
    # Use a generator for the split to make it reproducible
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))
    
    # Apply validation transform to the validation subset
    val_dataset.dataset.transform = val_transform

    # --- Dataloaders ---
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

    # --- Model Setup ---
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # Binary classification (0 or 1)
    model = model.to(DEVICE)
    
    # --- Early Stopping Setup ---
    # This object will track the best model across *both* phases
    early_stopper = EarlyStopping(patience=3,
                                  delta=0.001, # Min improvement
                                  save_path=BEST_MODEL_PATH,
                                  verbose=True)

    # --- Phase 1: Train Head Only ---
    print("\n===== Phase 1: Training the head =====")
    
    # Freeze all layers except the final one (fc)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    t_losses1, v_losses1, v_accs1, t_accs1 = train_one_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE,
        num_epochs=3,
        phase_name="head",
        early_stopper=early_stopper
    )

    # --- Phase 2: Fine-Tuning ---
    print("\nLoading best model before phase 2...")
    # Load the best model saved by EarlyStopper (from Phase 1)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    # Unfreeze the top layers ("layer4" and "fc") for fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optimizer for fine-tuning (only for unfrozen parameters, with a lower LR)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("\n===== Phase 2: Fine-tuning =====")
    t_losses2, v_losses2, v_accs2, t_accs2 = train_one_phase(
        model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE,
        num_epochs=7,
        phase_name="finetune",
        early_stopper=early_stopper
    )

    # --- Save History and Plot ---
    print("\n===== Generating Plots =====")
    history = {
        "train_loss": t_losses1 + t_losses2,
        "val_loss": v_losses1 + v_losses2,
        "train_acc": t_accs1 + t_accs2,
        "val_acc": v_accs1 + v_accs2
    }
    
    os.makedirs("../models", exist_ok=True)
    with open("../models/history.pkl", "wb") as f:
        pickle.dump(history, f)

    # Plot using the actual lengths of epochs that ran (due to early stopping)
    plot_training_history(history,
                          phase1_len=len(t_losses1),
                          phase2_len=len(t_losses2))

    print(f"\nTraining finished! History and plots saved.")
    print(f"The best model (based on val loss) is saved at: {BEST_MODEL_PATH}")
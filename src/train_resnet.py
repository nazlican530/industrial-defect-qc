import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time



# Reproducibility burada random sayı üreten tüm kütüphanelerde aynı seed'i ayarlayarak her çalıştırmada benzer sonuçlar almaya çalışıyoruz. Bu, modelin eğitim sürecinde rastgelelik içeren işlemlerin (örneğin, veri augmentasyonu, ağırlıkların başlangıç değeri) aynı şekilde gerçekleşmesini sağlar.

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main() -> None:
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dir = "data/NEU-DET/train/images"
    val_dir = "data/NEU-DET/validation/images"

    model_out_dir = Path("outputs/models")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    fig_out_dir = Path("outputs/figures")
    fig_out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_out_dir / "resnet50_best.pt"

    
    # Fair Train / Val Transforms
   
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10 # renk doygunluğunu rastgele değiştirme, modelin farklı aydınlatma koşullarında daha iyi genelleme yapmasına yardımcı olur.
        ),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.clamp(x + 0.01 * torch.randn_like(x), 0.0, 1.0)
        ),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

   
    # Dataset
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print("Classes:", train_dataset.classes)
    print("class_to_idx:", train_dataset.class_to_idx)
    num_classes = len(train_dataset.classes)

    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    
    # Class weights
    
    labels = [label for _, label in train_dataset.samples]
    counts = Counter(labels)

    class_weights = []
    for i in range(num_classes):
        class_weights.append(1.0 / counts[i])

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    
    # DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Model

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

   
    # Loss + Optimizer  
   
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    
    # Early stopping
    
    patience = 4
    no_improve = 0
    best_acc = 0.0

    
    # Epochs
    
    epochs = 10

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels_batch in loop:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            train_pred = logits.argmax(dim=1)
            train_total += labels_batch.size(0)
            train_correct += (train_pred == labels_batch).sum().item()

            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_train_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        train_acc = train_correct / train_total if train_total else 0.0
        train_accuracies.append(train_acc)

        
        # Validation
        
        model.eval()

        correct = 0
        total = 0
        preds_all = []
        labels_all = []
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)

                logits = model(images)
                loss = criterion(logits, labels_batch)
                val_loss_total += loss.item()

                pred = logits.argmax(dim=1)

                total += labels_batch.size(0)
                correct += (pred == labels_batch).sum().item()

                preds_all.extend(pred.cpu().numpy().tolist())
                labels_all.extend(labels_batch.cpu().numpy().tolist())

        acc = correct / total if total else 0.0
        avg_val_loss = val_loss_total / max(len(val_loader), 1)

        val_losses.append(avg_val_loss)
        val_accuracies.append(acc)

        epoch_time = time.time() - epoch_start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {acc:.4f}")
        print(f"Epoch Time: {minutes}m {seconds}s")

        print("\nValidation Classification Report:")
        print(
            classification_report(
                labels_all,
                preds_all,
                target_names=train_dataset.classes,
                digits=4,
                zero_division=0
            )
        )

        
        # Save best model
        
        if acc > best_acc:
            best_acc = acc
            no_improve = 0

            torch.save(
                {
                    "arch": "resnet50",
                    "model_state": model.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx,
                    "imagenet_mean": IMAGENET_MEAN,
                    "imagenet_std": IMAGENET_STD,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accuracies,
                    "val_accuracies": val_accuracies,
                },
                ckpt_path
            )

            print(f"Saved best model -> {ckpt_path}")

        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{patience}")

        
        # Early stopping
        
        if no_improve >= patience:
            print("Early stopping triggered")
            break

    print("\nTraining Complete")
    print("Best Validation Accuracy:", best_acc)

    
    # Plot losses
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet50 Training and Validation Losses")
    plt.legend()
    plt.tight_layout()

    loss_fig_path = fig_out_dir / "resnet50_loss_curve.png"
    plt.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved -> {loss_fig_path}")

    
    # Plot accuracies
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet50 Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()

    accuracy_fig_path = fig_out_dir / "resnet50_accuracy_curve.png"
    plt.savefig(accuracy_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Accuracy curve saved -> {accuracy_fig_path}")


if __name__ == "__main__":
    main()
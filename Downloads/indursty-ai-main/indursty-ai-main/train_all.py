import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet_model import EfficientNetB3Inspector, get_transforms
from models.vit_model import VisionTransformerInspector, get_vit_transforms
from models.yolo_model import YOLOv11Inspector

def train_efficientnet(dataset_dir, weights_dir, epochs=3, batch_size=16, lr=1e-4):
    print("\n--- Training EfficientNet-B3 Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    train_data = ImageFolder(root=dataset_dir / "classification" / "train", transform=train_transform)
    val_data   = ImageFolder(root=dataset_dir / "classification" / "val", transform=val_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = EfficientNetB3Inspector(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weights_dir / "efficientnet_b3.pth")
            
    elapsed = time.time() - start_time
    print(f"EfficientNet-B3 Training Complete in {elapsed:.1f}s. Best Val Acc: {best_acc*100:.2f}%")
    return best_acc

def train_vit(dataset_dir, weights_dir, epochs=3, batch_size=16, lr=1e-4):
    print("\n--- Training Vision Transformer (ViT-B/16) Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_transform, val_transform = get_vit_transforms()
    
    train_data = ImageFolder(root=dataset_dir / "classification" / "train", transform=train_transform)
    val_data   = ImageFolder(root=dataset_dir / "classification" / "val", transform=val_transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = VisionTransformerInspector(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weights_dir / "vit_b16.pth")
            
    elapsed = time.time() - start_time
    print(f"ViT-B/16 Training Complete in {elapsed:.1f}s. Best Val Acc: {best_acc*100:.2f}%")
    return best_acc

def train_yolo(dataset_dir, weights_dir, epochs=3):
    print("\n--- Training YOLOv11 Defect Detection Model ---")
    yaml_path = dataset_dir / "yolo" / "dataset.yaml"
    yolo_inspector = YOLOv11Inspector()
    
    results = yolo_inspector.train_model(
        data_yaml=str(yaml_path),
        epochs=epochs,
        imgsz=320,
        batch=16,
        project=str(weights_dir.parent / "runs"),
        name="yolo11_defect"
    )
    
    # Save best model to weights_dir
    best_pt = weights_dir.parent / "runs" / "yolo11_defect" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, weights_dir / "yolo11_defect.pt")
        print(f"YOLOv11 best model saved to {weights_dir / 'yolo11_defect.pt'}")
    return results

def main():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset" / "final_processed"
    weights_dir = base_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print("Processed dataset not found! Please run clean_dataset.py first.")
        return

    print("=" * 70)
    print("SMART MANUFACTURING AGENT - TRAINING ALL MODELS")
    print("=" * 70)
    
    eff_acc = train_efficientnet(dataset_dir, weights_dir, epochs=3)
    vit_acc = train_vit(dataset_dir, weights_dir, epochs=3)
    train_yolo(dataset_dir, weights_dir, epochs=3)
    
    print("\nAll Models Trained Successfully!")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from models.dual_net import DualTransferNet
from utils.dataset import get_data_loaders

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    acc = 100 * correct / total
    return running_loss / len(loader), acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    return running_loss / len(loader), acc

def main():
    # Configuration
    DATA_DIR = './data/lung_colon_image_set/' # Path to dataset
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Initialize TensorBoard Writer
    # Logs will be saved in 'runs/' directory
    writer = SummaryWriter('runs/lung_colon_experiment')
    
    # Load Data
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    print(f"Classes: {class_names}")
    
    # Initialize Model
    model = DualTransferNet(num_classes=len(class_names)).to(DEVICE)
    
    # Loss and Optimizer (Adam, as per manuscript)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training Loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Print to Console
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save best model and print best accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New Best Model Saved! (Best Val Acc: {best_val_acc:.2f}%)")
        else:
            print(f"Best Val Acc so far: {best_val_acc:.2f}%")

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
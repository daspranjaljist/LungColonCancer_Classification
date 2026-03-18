import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_data_loaders(data_dir, batch_size=32, img_size=768):
    # Data Augmentation as per Section 5.2
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        
        # Random Rotation
        transforms.RandomRotation(20),
        
        # Random Flipping
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # Random Zoom: Implemented via RandomResizedCrop
        # scale=(0.8, 1.2) means it crops an area between 80% to 120% of original size
        # and resizes it back to img_size, simulating Zoom In/Out.
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
        
        # Brightness Adjustment
        transforms.ColorJitter(brightness=0.1),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir)
    
    # Stratified Split (80% Train, 10% Val, 10% Test)
    targets = full_dataset.targets
    train_idx, temp_idx = train_test_split(
        np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[targets[i] for i in temp_idx], random_state=42
    )

    # Create subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Wrapper to apply transforms
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_set = TransformSubset(train_dataset, train_transform)
    val_set = TransformSubset(val_dataset, val_transform)
    test_set = TransformSubset(test_dataset, val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, full_dataset.classes
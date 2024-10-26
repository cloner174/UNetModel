import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import LungNoduleDataset
from Unet2d import Unet2D
from losses import CombinedLoss
import numpy as np
from sklearn.model_selection import train_test_split


def train_model(hyperparams, train_loader, val_loader):
    """
    Train the U-Net model with given hyperparameters.
    """
    
    learning_rate = hyperparams.get('learning_rate', 1e-4)
    batch_size = hyperparams.get('batch_size', 4)
    num_epochs = hyperparams.get('num_epochs', 50)
    weight_decay = hyperparams.get('weight_decay', 1e-5)
    base_filters = hyperparams.get('base_filters', 64)
    attention_params = hyperparams.get('attention_params', None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ct_dir = '/path/to/ct_scans'  
    mask_dir = '/path/to/masks'
    
    dataset = LungNoduleDataset(ct_dir, mask_dir)
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = Unet2D(input_channels=1, num_classes=1, base_filters=base_filters, attention_params=attention_params).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for ct_scans, masks in train_loader:
            ct_scans = ct_scans.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(ct_scans)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * ct_scans.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ct_scans, masks in val_loader:
                ct_scans = ct_scans.to(device)
                masks = masks.to(device)
                outputs = model(ct_scans)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * ct_scans.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return best_val_loss

#cloner174
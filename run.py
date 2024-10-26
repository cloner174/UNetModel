import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets import FullyAnnotatedDataset, WeaklyAnnotatedDataset
from main import train_model
from UNet import MultitaskAttentionUNet, MultitaskAttentionUNet_Pretrained
from utils import visualize_predictions



def go(base_dir, pro= False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    X = np.load(os.path.join(base_dir, 'X.npy')) 
    masks = np.load(os.path.join(base_dir, 'masks.npy')) 
    boxes = np.load(os.path.join(base_dir, 'y_centroids.npy'))
    y_class = np.load(os.path.join(base_dir, 'y_class.npy'))
    num_classes = int(y_class.max()) + 1
    X_weak = np.load(os.path.join(base_dir, 'images.npy'))
    y_train_weak = np.load(os.path.join(base_dir, 'labels.npy'))
    
    X_train, X_val, masks_train, masks_val, boxes_train, boxes_val, y_train, y_val = train_test_split(
        X , masks, boxes, y_class,
        test_size=0.2, random_state=42, shuffle=True
    )
    
    annotated_dataset = FullyAnnotatedDataset(X_train, masks_train, boxes_train, y_train, transform=None)
    weak_dataset = WeaklyAnnotatedDataset(X_weak, y_train_weak, transform=None)
    val_dataset = FullyAnnotatedDataset(X_val, masks_val, boxes_val, y_val, transform=None)
    
    annotated_loader = DataLoader(annotated_dataset, batch_size=64, shuffle=True, num_workers=4)
    weak_loader = DataLoader(weak_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    if not pro:
        model = MultitaskAttentionUNet(input_channels=1, num_classes=num_classes, bbox_size=4).to(device)
    else:
        model = MultitaskAttentionUNet_Pretrained(input_channels=1, num_classes=2, bbox_size=4).to(device)
    
    train_model(model, annotated_loader, weak_loader, val_loader, device, num_epochs=50, patience=10, base_dir=base_dir)
    
    model.load_state_dict(torch.load(os.path.join(base_dir, 'best_model.pth')))
    model.to(device)
    
    visualize_predictions(model, val_loader, device, num_samples=5, num_classes=2)
    

if __name__ == "__main__":
    bd = input('Please Insert The path to main dir of data: ')
    pro = input('Should use pretrained model? (yes/no): ')
    pro = True if pro.lower()== 'yes' else False
    go(bd, pro)

#cloner174
# pso_tuning.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyswarms.single.global_best import GlobalBestPSO
from torch.utils.data import DataLoader

from UNet import MultitaskAttentionUNet_Pretrained
from datasets import FullyAnnotatedDataset, WeaklyAnnotatedDataset
from main import train_model
from metrics import calculate_segmentation_metrics, calculate_classification_metrics, calculate_localization_metrics
from losses import CombinedLoss

# Define the hyperparameter bounds
hyperparameter_bounds = {
    'learning_rate': (1e-5, 1e-2),
    'weight_decay': (1e-6, 1e-3),
    'dropout_rate': (0.0, 0.5),
    'base_filters': (16, 128),  # Integer values
    # Attention parameters: Integer values
    'F_g1': (64, 512),
    'F_l1': (64, 512),
    'F_int1': (32, 256),
    'F_g2': (32, 256),
    'F_l2': (32, 256),
    'F_int2': (16, 128),
    'F_g3': (16, 128),
    'F_l3': (16, 128),
    'F_int3': (8, 64),
    'F_g4': (8, 64),
    'F_l4': (8, 64),
    'F_int4': (4, 32),
}

# Define the search space dimensions
dimensions = len(hyperparameter_bounds)
bounds = np.array([list(hyperparameter_bounds[key]) for key in hyperparameter_bounds]).T

def objective_function(hyperparameters,train_loader, weak_loader, val_loader):
    
    n_particles = hyperparameters.shape[0]
    loss = np.zeros(n_particles)
    for i in range(n_particles):
        params = hyperparameters[i]
        param_dict = {}
        idx = 0
        
        for key in hyperparameter_bounds.keys():
            value = params[idx]
            if 'F_' in key or key == 'base_filters':
                param_dict[key] = int(round(value))
            else:
                param_dict[key] = value
            idx += 1
        
        learning_rate = param_dict['learning_rate']
        weight_decay = param_dict['weight_decay']
        dropout_rate = param_dict['dropout_rate']
        base_filters = param_dict['base_filters']
        
        attention_params = {
            'F_g1': param_dict['F_g1'],
            'F_l1': param_dict['F_l1'],
            'F_int1': param_dict['F_int1'],
            'F_g2': param_dict['F_g2'],
            'F_l2': param_dict['F_l2'],
            'F_int2': param_dict['F_int2'],
            'F_g3': param_dict['F_g3'],
            'F_l3': param_dict['F_l3'],
            'F_int3': param_dict['F_int3'],
            'F_g4': param_dict['F_g4'],
            'F_l4': param_dict['F_l4'],
            'F_int4': param_dict['F_int4'],
        }
        
        model = MultitaskAttentionUNet_Pretrained(
            input_channels=1,
            num_classes=1,
            bbox_size=4,
            base_filters=base_filters,
            attention_params=attention_params,
            dropout_rate=dropout_rate
        )
        
        model = model.to(device)
        
        num_epochs = 5
        hyperparams = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'patience': 2
        }
        
        train_model(model, train_loader, weak_loader, val_loader, device, hyperparams, base_dir='./')
        val_loss = evaluate_model(model, val_loader, device)
        
        loss[i] = val_loss
    
    return loss


def evaluate_model(model, val_loader, device):
    
    model.eval()
    val_total_loss = 0.0
    val_total = 0
    criterion_seg = CombinedLoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_loc = nn.SmoothL1Loss()
    with torch.no_grad():
        for batch in val_loader:
            images, masks, labels, boxes = batch
            images = images.to(device)
            masks = masks.to(device).float()
            boxes = boxes.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs_seg, outputs_cls, outputs_loc = model(images)
            
            loss_seg = criterion_seg(outputs_seg, masks)
            loss_cls = criterion_cls(outputs_cls, labels)
            loss_loc = criterion_loc(outputs_loc, boxes)
            
            loss = loss_seg + loss_cls + loss_loc
            
            val_total_loss += loss.item() * images.size(0)
            val_total += images.size(0)
    
    avg_val_loss = val_total_loss / val_total
    return avg_val_loss


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=dimensions, options=options, bounds=bounds)
    
    best_cost, best_pos = optimizer.optimize(objective_function, iters=10)
    
    print("Best hyperparameters found:")
    idx = 0
    for key in hyperparameter_bounds.keys():
        value = best_pos[idx]
        if 'F_' in key or key == 'base_filters':
            value = int(round(value))
        print(f"{key}: {value}")
        idx += 1

#cloner174
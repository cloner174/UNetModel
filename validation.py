import torch
from metrics import compute_dice_coefficient, compute_iou
import numpy as np

def validate_model(model, val_loader, device):
    """
    Validate the model and compute metrics.
    """
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for ct_scans, masks in val_loader:
            ct_scans = ct_scans.to(device)
            masks = masks.to(device)
            outputs = model(ct_scans)
            outputs = (outputs > 0.5).float()
            dice = compute_dice_coefficient(outputs, masks)
            iou = compute_iou(outputs, masks)
            dice_scores.append(dice)
            iou_scores.append(iou)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    return avg_dice, avg_iou

#cloner174
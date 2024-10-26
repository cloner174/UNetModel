import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def calculate_segmentation_metrics(preds, masks):
    """
    Calculate Dice Coefficient and IoU for segmentation.
    preds: Tensor of shape [B, 1, H, W] after sigmoid activation
    masks: Tensor of shape [B, H, W]
    """
    preds = (preds > 0.5).float()
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum()
    dice = (2. * intersection) / (union + 1e-5)
    iou = intersection / (union - intersection + 1e-5)
    return dice, iou


def calculate_classification_metrics(preds, labels):
    preds = torch.sigmoid(preds).cpu().numpy()
    labels = labels.cpu().numpy()
    preds_binary = (preds > 0.5).astype(int)
    precision = precision_score(labels, preds_binary, average='binary', zero_division=0)
    recall = recall_score(labels, preds_binary, average='binary', zero_division=0)
    f1 = f1_score(labels, preds_binary, average='binary', zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, preds)
    except ValueError:
        roc_auc = 0.0
    return precision, recall, f1, roc_auc


def calculate_localization_metrics(preds, boxes):
    mae = torch.nn.functional.l1_loss(preds, boxes, reduction='mean').item()
    return mae


def compute_dice_coefficient(pred, target, smooth=1e-5):
    """
    Compute the Dice Coefficient.
    """
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def compute_iou(pred, target, smooth=1e-5):
    """
    Compute the Intersection over Union (IoU).
    """
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

#cloner174
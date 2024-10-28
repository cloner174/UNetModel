import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-5):
    """
    Compute Dice Loss.
    """
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice.mean()
    return loss


class CombinedLoss(nn.Module):
    """
    Combined BCE Loss and Dice Loss.
    """
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
    
    def forward(self, pred, target):
        """
        pred: Predicted segmentation logits, shape [B, 1, H, W]
        target: Ground truth segmentation masks, shape [B, 1, H, W]
        """
        bce_loss = self.bce(pred, target)
        d_loss = dice_loss(pred, target)
        total_loss = self.weight_bce * bce_loss + self.weight_dice * d_loss
        return total_loss
    
#cloner174
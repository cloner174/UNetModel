import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-5):
    """
    Compute Dice Loss.
    """
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
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        bce_loss = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        d_loss = dice_loss(pred, target)
        return bce_loss + d_loss
    
#cloner174
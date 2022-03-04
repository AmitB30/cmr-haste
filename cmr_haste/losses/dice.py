import torch
from torch import nn
from monai.networks.utils import one_hot


class DiceLoss(nn.Module):
    """Soft dice loss. Supports binary and multiclass segmentation."""

    def __init__(self, mode, from_logits=True, reduce=True):
        """
        Args:
            mode: Either 'binary' or 'multiclass'.
            from_logits: If True the inputs are assumed to be logits.
        """
        super().__init__()
        self.mode = mode
        self.from_logits = from_logits
        self.reduce = reduce
        self.eps = 1e-5

    def forward(self, y_pred, y_target):
        """
        Args:
            y_pred: Tensor of shape (B,C,H,W)
            y_target: Tensor of shape (B,1,H,W) or (B,C,H,W)
        """
        if self.from_logits:
            if self.mode == 'binary':
                y_pred = torch.sigmoid(y_pred)
            else:
                y_pred = torch.softmax(y_pred, 1)

        if self.mode == 'multiclass':
            y_target = one_hot(y_target, num_classes=y_pred.shape[1], dim=1)

        if y_target.shape != y_pred.shape:
            raise ValueError(f'Input ({y_pred.shape}) and target ({y_target.shape}) shapes do not match')

        dims = (2, 3)
        intersection = torch.sum(y_pred * y_target, dim=dims)
        denominator = torch.sum(y_pred, dim=dims) + torch.sum(y_target, dim=dims)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        loss = 1 - dice
        if self.reduce:
            loss = torch.mean(loss)
        return loss


def dice_score(y_pred, y_target):
    """Calculate Dice score. Both tensors must be binarized and of shape (C,H,W)."""
    y_pred = y_pred.float()
    y_target = y_target.float()

    if y_target.shape != y_pred.shape:
        raise ValueError("y_pred and y_target should have the same shapes")

    dims = (1, 2)
    intersection = torch.sum(y_pred * y_target, dim=dims)
    y_target_o = torch.sum(y_target, dim=dims)
    y_pred_o = torch.sum(y_pred, dim=dims)
    denominator = y_target_o + y_pred_o
    return torch.where(y_target_o > 0, (2.0 * intersection) / denominator, torch.tensor(float("nan"), device=y_target_o.device))

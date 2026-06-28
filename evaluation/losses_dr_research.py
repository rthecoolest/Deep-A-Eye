import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


class OrdinalDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        class_ids = torch.arange(
            logits.size(1),
            dtype=torch.float32,
            device=logits.device
        )
        expected_grade = (probs * class_ids.unsqueeze(0)).sum(dim=1)
        targets = targets.float()
        return F.mse_loss(expected_grade, targets, reduction="none")


class QualityAwareHybridLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, ordinal_lambda=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma)
        self.ordinal = OrdinalDistanceLoss()
        self.ordinal_lambda = ordinal_lambda

    def forward(self, logits, targets, quality_weights):
        focal_loss = self.focal(logits, targets)
        ordinal_loss = self.ordinal(logits, targets)
        total = focal_loss + self.ordinal_lambda * ordinal_loss
        total = total * quality_weights
        return total.mean()
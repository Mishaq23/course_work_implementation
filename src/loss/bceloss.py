import torch
from torch import nn


class BCELoss(nn.Module):
    """
    Binary cross-entropy loss for one-logit fake/real classification.
    """

    def __init__(self, pos_weight: float):
        super().__init__()

        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        logits = logits.float()
        labels = labels.float()

        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        return {"loss": self.loss(logits, labels)}

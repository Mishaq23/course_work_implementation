import torch
from torch import nn

from src.model.fusion import ClassifierHead
from src.model.video_encoder import VideoEncoder


class VideoOnlyModel(nn.Module):
    def __init__(
        self,
        video_input_dim: int | None = None,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        video_input_type: str = "features",
    ):
        super().__init__()
        self.video_encoder = VideoEncoder(
            input_dim=video_input_dim,
            output_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_type=video_input_type,
        )
        self.classifier = ClassifierHead(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, video: torch.Tensor, **batch) -> dict:
        video_embedding = self.video_encoder(video)
        logits = self.classifier(video_embedding)

        return {
            "logits": logits,
            "video_embedding": video_embedding,
        }

import torch
from torch import nn

from src.model.audio_encoder import AudioEncoder
from src.model.fusion import ClassifierHead, build_fusion
from src.model.video_encoder import VideoEncoder


class GatedAVModel(nn.Module):
    """
    Audio-video model with configurable fusion.

    Supported fusion_type values: average, concat, dwf, gmu, film, attention.
    """

    def __init__(
        self,
        audio_input_dim: int | None = None,
        video_input_dim: int | None = None,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        audio_input_type: str = "features",
        video_input_type: str = "features",
        fusion_type: str = "gmu",
        fusion_kwargs: dict | None = None,
    ):
        super().__init__()

        self.audio_encoder = AudioEncoder(
            input_dim=audio_input_dim,
            output_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_type=audio_input_type,
        )
        self.video_encoder = VideoEncoder(
            input_dim=video_input_dim,
            output_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_type=video_input_type,
        )

        fusion_kwargs = fusion_kwargs or {}
        self.fusion = build_fusion(
            fusion_type=fusion_type,
            input_dim=embedding_dim,
            **fusion_kwargs,
        )
        self.classifier = ClassifierHead(
            input_dim=self.fusion.output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        **batch,
    ) -> dict:
        audio_embedding = self.audio_encoder(audio)
        video_embedding = self.video_encoder(video)

        fusion_output = self.fusion(audio_embedding, video_embedding)
        logits = self.classifier(fusion_output["fused"])

        return {
            "logits": logits,
            "audio_embedding": audio_embedding,
            "video_embedding": video_embedding,
            "fused_embedding": fusion_output["fused"],
            "audio_weight": fusion_output["audio_weight"],
            "video_weight": fusion_output["video_weight"],
        }

import torch
from torch import nn

from src.model.audio_encoder import MLPProjector


class RawVideoEncoder(nn.Module):
    """
    Lightweight frame encoder with temporal average pooling.

    Input: [B, T, C, H, W]
    Output: [B, output_dim]
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = video.float()

        if video.max() > 1.0:
            video = video / 255.0

        if video.ndim != 5:
            raise ValueError(
                f"Expected video shape [B, T, C, H, W], got {video.shape}."
            )

        b, t, c, h, w = video.shape
        video = video.reshape(b * t, c, h, w)
        frames = self.frame_encoder(video).flatten(1)
        frames = frames.reshape(b, t, -1)
        return frames.mean(dim=1)


class VideoFeatureEncoder(nn.Module):
    """
    Project precomputed video embeddings, e.g. VideoMAE features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projector = MLPProjector(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim > 2:
            if video.shape[-1] == self.input_dim:
                reduce_dims = tuple(range(1, video.ndim - 1))
                video = video.mean(dim=reduce_dims)
            else:
                video = video.flatten(start_dim=1)

        return self.projector(video)


class VideoEncoder(nn.Module):
    """
    Selects raw-video or feature-level video encoding from config.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        input_type: str = "features",
    ):
        super().__init__()

        if input_type not in {"features", "raw"}:
            raise ValueError("input_type must be 'features' or 'raw'.")

        if input_type == "raw":
            self.encoder = RawVideoEncoder(output_dim=output_dim)
        else:
            if input_dim is None:
                raise ValueError("input_dim is required for feature video encoder.")

            self.encoder = VideoFeatureEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(video)

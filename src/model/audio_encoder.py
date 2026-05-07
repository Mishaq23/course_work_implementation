import torch
from torch import nn


class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim: int | None = None, dropout=0.1):
        super().__init__()

        hidden_dim = hidden_dim or output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.net(features.float())


class RawAudioEncoder(nn.Module):
    """
    Lightweight raw-waveform encoder.

    Input: [B, C, L] or [B, L]
    Output: [B, output_dim]
    """

    def __init__(self, output_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, output_dim, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, audio):
        audio = audio.float()

        if audio.ndim == 2:
            audio = audio.unsqueeze(1)

        if audio.ndim != 3:
            raise ValueError(f"Expected audio shape [B, C, L], got {audio.shape}.")

        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        return self.encoder(audio).flatten(1)


class AudioFeatureEncoder(nn.Module):
    """
    Project precomputed audio embeddings, e.g. WavLM features.
    """

    def __init__(self, input_dim, output_dim=256, hidden_dim: int | None = None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.projector = MLPProjector(input_dim, output_dim, hidden_dim, dropout)

    def forward(self, audio: torch.Tensor):
        if audio.ndim > 2:
            if audio.shape[-1] == self.input_dim:
                reduce_dims = tuple(range(1, audio.ndim - 1))
                audio = audio.mean(dim=reduce_dims)
            else:
                audio = audio.flatten(start_dim=1)

        return self.projector(audio)


class AudioEncoder(nn.Module):
    """
    Selects raw-waveform or feature-level audio encoding from input shape.
    """

    def __init__(self, input_dim, output_dim=256, hidden_dim=None, 
                 dropout=0.1, input_type="features"):
        super().__init__()

        if input_type not in {"features", "raw"}:
            raise ValueError("input_type must be 'features' or 'raw'.")

        self.input_type = input_type

        if input_type == "raw":
            self.encoder = RawAudioEncoder(output_dim=output_dim)
        else:
            if input_dim is None:
                raise ValueError("input_dim is required for feature audio encoder.")

            self.encoder = AudioFeatureEncoder(input_dim, output_dim, hidden_dim, dropout)

    def forward(self, audio):
        return self.encoder(audio)

import torch
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor):
        return self.net(features).squeeze(-1)


class ConcatFusion(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = input_dim * 2

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        return {
            "fused": torch.cat([audio, video], dim=-1),
            "audio_weight": None,
            "video_weight": None
        }


class AverageFusion(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        return {
            "fused": 0.5 * (audio + video),
            "audio_weight": torch.full_like(audio[:, :1], 0.5),
            "video_weight": torch.full_like(video[:, :1], 0.5)
        }


class DynamicWeightFusion(nn.Module):
    """
    DWF-style adaptive modality weighting from audio/video embeddings.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.output_dim = input_dim
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        weights = torch.softmax(
            self.weight_net(torch.cat([audio, video], dim=-1)),
            dim=-1,
        )
        audio_weight = weights[:, 0:1]
        video_weight = weights[:, 1:2]
        fused = audio_weight * audio + video_weight * video

        return {
            "fused": fused,
            "audio_weight": audio_weight,
            "video_weight": video_weight
        }


class GMUFusion(nn.Module):
    """
    Gated Multimodal Unit.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = input_dim
        self.audio_proj = nn.Linear(input_dim, input_dim)
        self.video_proj = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(input_dim * 2, input_dim)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        audio_hidden = torch.tanh(self.audio_proj(audio))
        video_hidden = torch.tanh(self.video_proj(video))
        gate = torch.sigmoid(self.gate(torch.cat([audio, video], dim=-1)))
        fused = gate * audio_hidden + (1.0 - gate) * video_hidden

        return {
            "fused": fused,
            "audio_weight": gate.mean(dim=-1, keepdim=True),
            "video_weight": (1.0 - gate).mean(dim=-1, keepdim=True)
        }


class FiLMFusion(nn.Module):
    """
    Feature-wise linear modulation of video features by audio context.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = input_dim * 2
        self.audio_to_video = nn.Linear(input_dim, input_dim * 2)
        self.video_to_audio = nn.Linear(input_dim, input_dim * 2)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> dict:
        video_gamma, video_beta = self.audio_to_video(audio).chunk(2, dim=-1)
        audio_gamma, audio_beta = self.video_to_audio(video).chunk(2, dim=-1)

        mod_video = video * (1.0 + torch.tanh(video_gamma)) + video_beta
        mod_audio = audio * (1.0 + torch.tanh(audio_gamma)) + audio_beta
        fused = torch.cat([mod_audio, mod_video], dim=-1)

        return {
            "fused": fused,
            "audio_weight": None,
            "video_weight": None
        }


class AttentionFusion(nn.Module):
    """
    Small cross-modal attention block for two modality tokens.
    """

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.output_dim = input_dim * 2
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        tokens = torch.stack([audio, video], dim=1)
        attended, attn_weights = self.attention(tokens, tokens, tokens, need_weights=True)
        attended = self.norm(attended + tokens)
        fused = attended.flatten(start_dim=1)

        modality_weights = attn_weights.mean(dim=1)
        return {
            "fused": fused,
            "audio_weight": modality_weights[:, 0:1],
            "video_weight": modality_weights[:, 1:2]
        }


FUSION_REGISTRY = {
    "average": AverageFusion,
    "concat": ConcatFusion,
    "dwf": DynamicWeightFusion,
    "gmu": GMUFusion,
    "film": FiLMFusion,
    "attention": AttentionFusion
}


def build_fusion(fusion_type: str, input_dim: int, **kwargs) -> nn.Module:
    if fusion_type not in FUSION_REGISTRY:
        available = ", ".join(sorted(FUSION_REGISTRY))
        raise ValueError(f"Unknown fusion_type={fusion_type}. Available: {available}.")

    return FUSION_REGISTRY[fusion_type](input_dim=input_dim, **kwargs)

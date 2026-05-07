import torch
from torch import nn

from src.model.audio_encoder import AudioEncoder
from src.model.fusion import ClassifierHead


class AudioOnlyModel(nn.Module):
    def __init__(self, audio_input_dim=None, embedding_dim=256, 
                 hidden_dim=256, dropout=0.2, audio_input_type="features"):
        super().__init__()
        self.audio_encoder = AudioEncoder(audio_input_dim, embedding_dim, 
                                          hidden_dim, dropout, audio_input_type)
        self.classifier = ClassifierHead(embedding_dim, hidden_dim, dropout)

    def forward(self, audio, **batch):
        audio_embedding = self.audio_encoder(audio)
        logits = self.classifier(audio_embedding)

        return {
            "logits": logits,
            "audio_embedding": audio_embedding,
        }

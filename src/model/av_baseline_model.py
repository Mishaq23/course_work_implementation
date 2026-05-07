from src.model.gated_av_model import GatedAVModel


class AVBaselineModel(GatedAVModel):
    """
    Backward-compatible AV baseline.

    Defaults to raw audio/video inputs and concat fusion, matching the old
    baseline behavior while reusing the new model block.
    """

    def __init__(self, hidden_dim=128, dropout=0.3,
                 embedding_dim=None, fusion_type="concat"):
        embedding_dim = embedding_dim or hidden_dim

        super().__init__(
            audio_input_dim=None,
            video_input_dim=None,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            audio_input_type="raw",
            video_input_type="raw",
            fusion_type=fusion_type,
        )

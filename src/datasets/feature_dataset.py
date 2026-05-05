from pathlib import Path

import torch

from src.datasets.base_dataset import BaseDataset


class AVFeatureDataset(BaseDataset):
    """
    Universal feature-level dataset.

    Used for training/evaluation on precomputed audio/video embeddings.
    """

    def __init__(
        self,
        features_dir: str | Path,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms=None,
    ):
        self.features_dir = Path(features_dir)

        self.audio_features = torch.load(self.features_dir / "audio_features.pt")
        self.video_features = torch.load(self.features_dir / "video_features.pt")
        self.labels = torch.load(self.features_dir / "labels.pt")
        self.meta = torch.load(self.features_dir / "meta.pt")

        index = self._create_index()

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _create_index(self) -> list[dict]:
        index = []

        for sample_id in sorted(self.labels.keys()):
            meta = self.meta[sample_id]

            index.append(
                {
                    "sample_id": sample_id,
                    "path": sample_id,
                    "label": self.labels[sample_id],
                    "dataset": meta.get("dataset", "unknown"),
                    "fake_type": meta.get("fake_type", "unknown"),
                    "degradation": meta.get("degradation", "clean"),
                    "audio_degradation": meta.get("audio_degradation", "clean"),
                    "video_degradation": meta.get("video_degradation", "clean"),
                }
            )

        return index

    def __getitem__(self, ind: int) -> dict:
        item = self._index[ind]
        sample_id = item["sample_id"]

        instance_data = {
            "sample_id": sample_id,
            "dataset": item["dataset"],
            "audio": self.audio_features[sample_id].float(),
            "video": self.video_features[sample_id].float(),
            "labels": torch.tensor(item["label"], dtype=torch.float32),
            "fake_type": item["fake_type"],
            "degradation": item["degradation"],
            "audio_degradation": item["audio_degradation"],
            "video_degradation": item["video_degradation"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

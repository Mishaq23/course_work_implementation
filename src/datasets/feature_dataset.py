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
        self._validate_feature_dir()

        self.audio_features = torch.load(
            self.features_dir / "audio_features.pt",
            map_location="cpu",
        )
        self.video_features = torch.load(
            self.features_dir / "video_features.pt",
            map_location="cpu",
        )
        self.labels = torch.load(self.features_dir / "labels.pt", map_location="cpu")
        self.meta = torch.load(self.features_dir / "meta.pt", map_location="cpu")
        self._validate_feature_keys()

        index = self._create_index()

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _validate_feature_dir(self) -> None:
        required_files = [
            "audio_features.pt",
            "video_features.pt",
            "labels.pt",
            "meta.pt",
        ]
        missing = [
            file_name for file_name in required_files
            if not (self.features_dir / file_name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing feature files in {self.features_dir}: {', '.join(missing)}"
            )

    def _validate_feature_keys(self) -> None:
        audio_ids = set(self.audio_features.keys())
        video_ids = set(self.video_features.keys())
        label_ids = set(self.labels.keys())
        meta_ids = set(self.meta.keys())

        common_ids = audio_ids & video_ids & label_ids & meta_ids
        if len(common_ids) == 0:
            raise ValueError(
                f"No common sample ids found across feature files in {self.features_dir}."
            )

        mismatched_sources = []
        if audio_ids != common_ids:
            mismatched_sources.append("audio_features")
        if video_ids != common_ids:
            mismatched_sources.append("video_features")
        if label_ids != common_ids:
            mismatched_sources.append("labels")
        if meta_ids != common_ids:
            mismatched_sources.append("meta")

        if mismatched_sources:
            raise ValueError(
                "Feature files contain mismatched sample ids for: "
                + ", ".join(mismatched_sources)
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

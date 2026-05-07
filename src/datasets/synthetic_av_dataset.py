import torch

from src.datasets.base_dataset import BaseDataset


class SyntheticAVDataset(BaseDataset):
    """
    Small synthetic dataset for smoke tests and example configs.
    """

    def __init__(
        self,
        dataset_length: int = 16,
        audio_shape: list[int] | tuple[int, ...] = (1024,),
        video_shape: list[int] | tuple[int, ...] = (768,),
        positive_fraction: float = 0.5,
        seed: int = 42,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms=None,
    ):
        generator = torch.Generator().manual_seed(seed)
        self.audio_shape = tuple(audio_shape)
        self.video_shape = tuple(video_shape)

        dataset_length = int(dataset_length)
        n_positive = int(round(dataset_length * positive_fraction))
        labels = [1] * n_positive + [0] * max(0, dataset_length - n_positive)

        self.audio_data = {}
        self.video_data = {}
        index = []

        for sample_idx, label in enumerate(labels):
            sample_id = f"synthetic_{sample_idx:05d}"
            self.audio_data[sample_id] = torch.randn(
                self.audio_shape,
                generator=generator,
            )
            self.video_data[sample_id] = torch.randn(
                self.video_shape,
                generator=generator,
            )
            index.append(
                {
                    "sample_id": sample_id,
                    "path": sample_id,
                    "label": float(label),
                    "dataset": "synthetic_av",
                    "fake_type": "synthetic_fake" if label == 1 else "synthetic_real",
                    "degradation": "clean",
                    "audio_degradation": "clean",
                    "video_degradation": "clean",
                }
            )

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def __getitem__(self, ind: int):
        item = self._index[ind]
        sample_id = item["sample_id"]
        instance_data = {
            "sample_id": sample_id,
            "dataset": item["dataset"],
            "audio": self.audio_data[sample_id].clone(),
            "video": self.video_data[sample_id].clone(),
            "labels": torch.tensor(item["label"], dtype=torch.float32),
            "fake_type": item["fake_type"],
            "degradation": item["degradation"],
            "audio_degradation": item["audio_degradation"],
            "video_degradation": item["video_degradation"],
        }
        return self.preprocess_data(instance_data)

import torch
import torchvision.io as io

from src.datasets.base_dataset import BaseDataset


class AVRawDataset(BaseDataset):
    """
    Universal audio-video raw dataset.

    It receives an index and returns:
    video tensor, audio tensor, label, metadata.
    """

    def __init__(
        self,
        index: list[dict],
        num_frames: int = 16,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms=None,
    ):
        self.num_frames = num_frames

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def __getitem__(self, ind: int) -> dict:
        item = self._index[ind]
        video, audio, info = self.load_object(item["path"])

        video = self._sample_frames(video)
        video = self._preprocess_video(video)

        instance_data = {
            "sample_id": item["sample_id"],
            "dataset": item.get("dataset", "unknown"),
            "video": video,
            "audio": audio,
            "labels": torch.tensor(item["label"], dtype=torch.float32),
            "fake_type": item.get("fake_type", "unknown"),
            "path": item["path"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def load_object(self, path: str):
        video, audio, info = io.read_video(path, pts_unit="sec")
        return video, audio, info

    def _sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        total_frames = video.shape[0]

        if total_frames == 0:
            raise RuntimeError("Empty video")

        indices = torch.linspace(
            0,
            total_frames - 1,
            steps=self.num_frames,
        ).long()

        return video[indices]

    def _preprocess_video(self, video: torch.Tensor) -> torch.Tensor:
        # [T, H, W, C] uint8 -> [T, C, H, W] float [0, 1]
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)
        return video

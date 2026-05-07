import torch
import torch.nn.functional as F
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
        video_size: int | tuple[int, int] | None = 224,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms=None,
    ):
        self.num_frames = num_frames
        self.video_size = video_size

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
        audio = self._preprocess_audio(audio)

        instance_data = {
            "sample_id": item["sample_id"],
            "dataset": item.get("dataset", "unknown"),
            "video": video,
            "audio": audio,
            "labels": torch.tensor(item["label"], dtype=torch.float32),
            "fake_type": item.get("fake_type", "unknown"),
            "degradation": item.get("degradation", "clean"),
            "audio_degradation": item.get("audio_degradation", "clean"),
            "video_degradation": item.get("video_degradation", "clean"),
            "audio_sample_rate": info.get("audio_fps", 0),
            "video_fps": info.get("video_fps", 0),
            "path": item["path"],
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def load_object(self, path: str):
        try:
            video, audio, info = io.read_video(path, pts_unit="sec")
        except Exception as exc:
            raise RuntimeError(f"Failed to read audio/video file: {path}") from exc

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

        if self.video_size is not None:
            if isinstance(self.video_size, int):
                size = (self.video_size, self.video_size)
            else:
                size = self.video_size

            video = F.interpolate(
                video,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return video

    def _preprocess_audio(self, audio: torch.Tensor | None) -> torch.Tensor:
        if audio is None:
            return torch.zeros(1, 1)

        audio = audio.float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Some backends return audio as [num_samples, num_channels].
        if audio.ndim == 2 and audio.shape[0] > audio.shape[1] and audio.shape[1] <= 8:
            audio = audio.transpose(0, 1)

        if audio.numel() == 0:
            return torch.zeros(1, 1)

        return audio

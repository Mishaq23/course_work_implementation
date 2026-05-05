import torch
import torch.nn.functional as F


def _pad_audio_batch(audio_items: list[torch.Tensor]) -> torch.Tensor:
    """
    Pads audio tensors to the maximum audio length in the batch.

    Expected audio shape from torchvision.io.read_video:
        [channels, num_audio_samples]

    Returns:
        [batch_size, channels, max_num_audio_samples]
    """
    processed = []

    max_channels = 1
    max_length = 1

    for audio in audio_items:
        if audio is None:
            audio = torch.zeros(1, 1)

        if not torch.is_tensor(audio):
            audio = torch.tensor(audio)

        audio = audio.float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.numel() == 0:
            audio = torch.zeros(1, 1)

        max_channels = max(max_channels, audio.shape[0])
        max_length = max(max_length, audio.shape[1])
        processed.append(audio)

    padded = []

    for audio in processed:
        channels, length = audio.shape

        if channels < max_channels:
            channel_pad = torch.zeros(
                max_channels - channels,
                length,
                dtype=audio.dtype,
            )
            audio = torch.cat([audio, channel_pad], dim=0)

        if length < max_length:
            audio = F.pad(audio, (0, max_length - length))

        padded.append(audio)

    return torch.stack(padded, dim=0)


def collate_fn(dataset_items: list[dict]) -> dict:
    """
    Collate function for audio-video samples.

    It stacks fixed-size video tensors and pads variable-length audio tensors.
    """
    result_batch = {}

    if "video" in dataset_items[0]:
        result_batch["video"] = torch.stack(
            [item["video"] for item in dataset_items],
            dim=0,
        )

    if "audio" in dataset_items[0]:
        result_batch["audio"] = _pad_audio_batch(
            [item["audio"] for item in dataset_items]
        )

    result_batch["labels"] = torch.stack(
        [
            item["labels"]
            if torch.is_tensor(item["labels"])
            else torch.tensor(item["labels"], dtype=torch.float32)
            for item in dataset_items
        ],
        dim=0,
    ).float()

    if "sample_id" in dataset_items[0]:
        result_batch["sample_id"] = [
            item["sample_id"] for item in dataset_items
        ]

    if "dataset" in dataset_items[0]:
        result_batch["dataset"] = [
            item["dataset"] for item in dataset_items
        ]

    if "fake_type" in dataset_items[0]:
        result_batch["fake_type"] = [
            item["fake_type"] for item in dataset_items
        ]

    if "path" in dataset_items[0]:
        result_batch["path"] = [
            item["path"] for item in dataset_items
        ]

    return result_batch

import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    if "audio" in dataset_items[0]:
        result_batch["audio"] = torch.stack(
            [item["audio"] for item in dataset_items],
            dim=0,
        )

    if "video" in dataset_items[0]:
        result_batch["video"] = torch.stack(
            [item["video"] for item in dataset_items],
            dim=0,
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
        result_batch["sample_id"] = [item["sample_id"] for item in dataset_items]

    if "fake_type" in dataset_items[0]:
        result_batch["fake_type"] = [item["fake_type"] for item in dataset_items]

    if "path" in dataset_items[0]:
        result_batch["path"] = [item["path"] for item in dataset_items]

    return result_batch

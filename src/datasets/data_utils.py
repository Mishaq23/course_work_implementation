import logging
from itertools import repeat

import torch
from hydra.utils import instantiate
from torch.utils.data import WeightedRandomSampler

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

logger = logging.getLogger(__name__)


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    if batch_transforms is None:
        return

    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def build_balanced_train_sampler(dataset):
    """
    Build a weighted sampler that approximately balances classes.

    The sampler uses inverse class frequency weights and samples with
    replacement. This is useful for strongly imbalanced binary datasets such
    as FakeAVCeleb after collapsing all fake types into one positive label.
    """
    if not hasattr(dataset, "get_labels"):
        raise TypeError(
            "Weighted sampling requires the dataset to implement get_labels()."
        )

    labels = torch.tensor(dataset.get_labels(), dtype=torch.long)
    if labels.numel() == 0:
        raise ValueError("Cannot build weighted sampler for an empty dataset.")

    unique_labels = torch.unique(labels)
    if unique_labels.numel() < 2:
        raise ValueError(
            "Weighted sampling requires at least two classes in the train split."
        )

    class_counts = torch.bincount(labels)
    if torch.any(class_counts == 0):
        raise ValueError(
            "Weighted sampling requires every present class index to have "
            "at least one sample."
        )

    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]

    logger.info(
        "Using weighted train sampler with class counts: %s",
        class_counts.tolist(),
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        if len(dataset) == 0:
            raise ValueError(f"Dataset partition '{dataset_partition}' is empty.")

        if dataset_partition == "train" and config.dataloader.batch_size > len(dataset):
            raise ValueError(
                f"Train batch size ({config.dataloader.batch_size}) cannot be larger "
                f"than train dataset length ({len(dataset)}), because drop_last=True "
                "would yield zero batches."
            )

        sampler = None
        shuffle = dataset_partition == "train"
        if dataset_partition == "train" and config.trainer.get(
            "use_weighted_sampler", False
        ):
            sampler = build_balanced_train_sampler(dataset)
            shuffle = False

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=shuffle,
            sampler=sampler,
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms

import logging
import random
from pathlib import Path

from tqdm.auto import tqdm

from src.datasets.av_raw_dataset import AVRawDataset
from src.datasets.label_inference import infer_fakeavceleb_label
from src.utils.io_utils import read_json, write_json

logger = logging.getLogger(__name__)


class FakeAVCelebDataset(AVRawDataset):
    """
    Dataset class for FakeAVCeleb.

    It searches for all .mp4 files inside root_dir, infers labels from file paths,
    creates a full index, and then creates deterministic train/val/test splits.
    """

    def __init__(
        self,
        root_dir: str | Path,
        index_dir: str | Path | None = None,
        num_frames: int = 16,
        video_size: int | tuple[int, int] | None = 224,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms=None,
        name: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 42,
        rebuild_index: bool = False,
        video_extensions: tuple[str, ...] = (".mp4",),
        *args,
        **kwargs,
    ):
        self.root_dir = Path(root_dir)
        self.index_dir = Path(index_dir) if index_dir is not None else self.root_dir
        self.video_extensions = video_extensions
        self.rebuild_index = rebuild_index

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"FakeAVCeleb root_dir does not exist: {self.root_dir}"
            )

        self.index_dir.mkdir(parents=True, exist_ok=True)

        if name not in ["train", "val", "test", "full"]:
            raise ValueError(
                f"Unknown split name: {name}. "
                f"Expected one of: train, val, test, full."
            )

        self.name = name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        self._check_split_ratios()

        index_path = self.index_dir / f"index_{name}.json"

        if self.rebuild_index or not index_path.exists():
            self._create_all_indices()

        index = read_json(str(index_path))

        super().__init__(
            index=index,
            num_frames=num_frames,
            video_size=video_size,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
            *args,
            **kwargs,
        )

    def _check_split_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must be equal to 1. "
                f"Got {total}."
            )

    def _create_all_indices(self) -> None:
        full_index_path = self.index_dir / "index_full.json"
        train_index_path = self.index_dir / "index_train.json"
        val_index_path = self.index_dir / "index_val.json"
        test_index_path = self.index_dir / "index_test.json"

        if full_index_path.exists() and not self.rebuild_index:
            full_index = read_json(str(full_index_path))
        else:
            full_index = self._create_full_index()
            write_json(full_index, str(full_index_path))

        train_index, val_index, test_index = self._split_index(full_index)

        write_json(train_index, str(train_index_path))
        write_json(val_index, str(val_index_path))
        write_json(test_index, str(test_index_path))

    def _create_full_index(self) -> list[dict]:
        video_paths = sorted(
            path
            for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.video_extensions
        )

        if len(video_paths) == 0:
            raise RuntimeError(
                f"No video files with extensions {self.video_extensions} "
                f"found in {self.root_dir}"
            )

        index = []

        logger.info("Creating FakeAVCeleb full index")

        for idx, video_path in enumerate(tqdm(video_paths)):
            label, fake_type = infer_fakeavceleb_label(video_path)
            relative_path = video_path.relative_to(self.root_dir)

            index.append(
                {
                    "sample_id": f"fakeavceleb_{idx:06d}",
                    "path": str(video_path),
                    "relative_path": str(relative_path),
                    "label": label,
                    "dataset": "fakeavceleb",
                    "fake_type": fake_type,
                    "degradation": "clean",
                    "audio_degradation": "clean",
                    "video_degradation": "clean",
                }
            )

        return index

    def _split_index(self, full_index: list[dict]):
        rng = random.Random(self.split_seed)

        real_samples = [item for item in full_index if item["label"] == 0]
        fake_samples = [item for item in full_index if item["label"] == 1]

        rng.shuffle(real_samples)
        rng.shuffle(fake_samples)

        real_train, real_val, real_test = self._split_group(real_samples)
        fake_train, fake_val, fake_test = self._split_group(fake_samples)

        train_index = real_train + fake_train
        val_index = real_val + fake_val
        test_index = real_test + fake_test

        rng.shuffle(train_index)
        rng.shuffle(val_index)
        rng.shuffle(test_index)

        logger.info(
            "FakeAVCeleb split sizes: "
            f"train={len(train_index)}, "
            f"val={len(val_index)}, "
            f"test={len(test_index)}"
        )

        return train_index, val_index, test_index

    def _split_group(self, samples: list[dict]):
        n_total = len(samples)

        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val :]

        return train_samples, val_samples, test_samples

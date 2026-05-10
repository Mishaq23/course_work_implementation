import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from src.datasets.av_raw_dataset import AVRawDataset
from src.datasets.label_inference import infer_fakeavceleb_label
from src.utils.io_utils import read_json, write_json

logger = logging.getLogger(__name__)


SPLIT_SCHEMA_VERSIONS = {
    "sample": "v1",
    "id_tuple": "v1",
    "id_component": "v2",
}


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
        split_strategy: str = "id_component",
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
        self.split_strategy = split_strategy
        self._id_pattern = re.compile(r"id\d+")

        self._check_split_ratios()
        self._check_split_strategy()

        index_path = self.index_dir / self._split_index_filename(name)

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

    def _check_split_strategy(self) -> None:
        valid_strategies = {"sample", "id_tuple", "id_component"}
        if self.split_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown split_strategy={self.split_strategy!r}. "
                f"Expected one of {sorted(valid_strategies)}."
            )

    def _split_index_filename(self, split_name: str) -> str:
        if split_name == "full":
            return "index_full.json"
        schema_version = SPLIT_SCHEMA_VERSIONS[self.split_strategy]
        return f"index_{self.split_strategy}_{schema_version}_{split_name}.json"

    def _create_all_indices(self) -> None:
        full_index_path = self.index_dir / self._split_index_filename("full")
        train_index_path = self.index_dir / self._split_index_filename("train")
        val_index_path = self.index_dir / self._split_index_filename("val")
        test_index_path = self.index_dir / self._split_index_filename("test")

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
                    "group_key": self._build_group_key_from_path(relative_path),
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
        if self.split_strategy == "sample":
            return self._split_index_samplewise(full_index)

        return self._split_index_groupwise(full_index)

    def _split_index_samplewise(self, full_index: list[dict]):
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

    def _split_index_groupwise(self, full_index: list[dict]):
        rng = random.Random(self.split_seed)
        group_to_items = defaultdict(list)
        label_totals = Counter()

        if hasattr(self, "_component_group_cache"):
            delattr(self, "_component_group_cache")

        for item in full_index:
            group_key = self._get_group_key(item, full_index)
            group_to_items[group_key].append(item)
            label_totals[int(item["label"])] += 1

        group_records = []
        for group_key, items in group_to_items.items():
            label_counts = Counter(int(item["label"]) for item in items)
            group_records.append(
                {
                    "group_key": group_key,
                    "items": items,
                    "size": len(items),
                    "label_counts": label_counts,
                }
            )

        rng.shuffle(group_records)
        group_records.sort(
            key=lambda record: (record["size"], tuple(sorted(record["label_counts"].items()))),
            reverse=True,
        )

        split_names = ["train", "val", "test"]
        split_ratios = {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }

        targets = {
            split_name: {
                "size": len(full_index) * split_ratios[split_name],
                "label_counts": {
                    label: label_totals[label] * split_ratios[split_name]
                    for label in label_totals
                },
            }
            for split_name in split_names
        }
        split_state = {
            split_name: {
                "items": [],
                "size": 0,
                "label_counts": Counter(),
            }
            for split_name in split_names
        }

        for record in group_records:
            best_split = min(
                split_names,
                key=lambda split_name: self._group_assignment_score(
                    state=split_state[split_name],
                    target=targets[split_name],
                    record=record,
                    all_labels=label_totals.keys(),
                ),
            )

            annotated_items = []
            for item in record["items"]:
                annotated_item = dict(item)
                annotated_item["split_group_key"] = record["group_key"]
                annotated_items.append(annotated_item)

            split_state[best_split]["items"].extend(annotated_items)
            split_state[best_split]["size"] += record["size"]
            split_state[best_split]["label_counts"].update(record["label_counts"])

        for split_name in split_names:
            rng.shuffle(split_state[split_name]["items"])

        logger.info(
            "FakeAVCeleb group-wise split sizes: train=%d, val=%d, test=%d | groups=%d",
            len(split_state["train"]["items"]),
            len(split_state["val"]["items"]),
            len(split_state["test"]["items"]),
            len(group_records),
        )

        return (
            split_state["train"]["items"],
            split_state["val"]["items"],
            split_state["test"]["items"],
        )

    def _group_assignment_score(self, state, target, record, all_labels):
        projected_size = state["size"] + record["size"]
        size_target = max(float(target["size"]), 1.0)
        score = abs(projected_size - target["size"]) / size_target

        for label in all_labels:
            projected_label_count = state["label_counts"][label] + record["label_counts"][label]
            label_target = float(target["label_counts"].get(label, 0.0))
            denom = max(label_target, 1.0)
            score += abs(projected_label_count - label_target) / denom

        # Small penalty for overfilling a split; encourages earlier use of still-empty splits.
        if projected_size > target["size"]:
            score += (projected_size - target["size"]) / size_target

        return score

    def _build_group_key_from_path(self, path: str | Path) -> str:
        path_str = str(path).lower()
        id_tokens = sorted(set(self._id_pattern.findall(path_str)))
        if id_tokens:
            return "::".join(id_tokens)
        return path_str

    def _extract_id_tokens(self, item: dict) -> tuple[str, ...]:
        relative_path = item.get("relative_path", item.get("path", ""))
        return tuple(sorted(set(self._id_pattern.findall(str(relative_path).lower()))))

    def _build_component_group_map(self, full_index: list[dict]) -> dict[str, str]:
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        sample_tokens: dict[str, tuple[str, ...]] = {}
        for item in full_index:
            sample_id = item["sample_id"]
            tokens = self._extract_id_tokens(item)
            sample_tokens[sample_id] = tokens
            if len(tokens) >= 2:
                anchor = tokens[0]
                for token in tokens[1:]:
                    union(anchor, token)

        component_map: dict[str, str] = {}
        for item in full_index:
            sample_id = item["sample_id"]
            tokens = sample_tokens[sample_id]
            if tokens:
                roots = sorted({find(token) for token in tokens})
                component_map[sample_id] = "component::" + "::".join(roots)
            else:
                relative_path = item.get("relative_path", item.get("path", ""))
                component_map[sample_id] = "path::" + str(relative_path).lower()

        return component_map

    def _get_group_key(self, item: dict, full_index: list[dict] | None = None) -> str:
        if self.split_strategy == "id_component":
            if not hasattr(self, "_component_group_cache"):
                if full_index is None:
                    raise ValueError("full_index is required for id_component grouping.")
                self._component_group_cache = self._build_component_group_map(full_index)
            return self._component_group_cache[item["sample_id"]]

        if self.split_strategy == "id_tuple":
            if "group_key" in item:
                return item["group_key"]

            relative_path = item.get("relative_path", item.get("path", ""))
            return self._build_group_key_from_path(relative_path)

        if "group_key" in item:
            return item["group_key"]

        relative_path = item.get("relative_path", item.get("path", ""))
        return self._build_group_key_from_path(relative_path)

    def _split_group(self, samples: list[dict]):
        n_total = len(samples)

        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val :]

        return train_samples, val_samples, test_samples

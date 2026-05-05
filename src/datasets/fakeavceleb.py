from pathlib import Path

from tqdm.auto import tqdm

from src.datasets.av_raw_dataset import AVRawDataset
from src.datasets.label_inference import infer_fakeavceleb_label
from src.utils.io_utils import read_json, write_json


class FakeAVCelebDataset(AVRawDataset):
    """
    Dataset class for FakeAVCeleb.

    The dataset searches for all .mp4 files inside root_dir,
    infers labels from file paths, and creates an index with
    metadata for each video sample.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int = 16,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms=None,
        name: str = "full",
        *args,
        **kwargs,
    ):
        """
        Args:
            root_dir (str | Path): path to FakeAVCeleb dataset root directory.
            num_frames (int): number of frames sampled from each video.
            limit (int | None): optional limit on dataset size.
            shuffle_index (bool): whether to shuffle index.
            instance_transforms: transforms applied to one dataset sample.
            name (str): partition/index name, for example "train", "val", "test", "full".
        """
        self.root_dir = Path(root_dir)

        index_path = self.root_dir / f"index_{name}.json"

        # each dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, dataset name, fake type, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(index_path)

        super().__init__(
            index=index,
            num_frames=num_frames,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
            *args,
            **kwargs,
        )

    def _create_index(self, index_path: str | Path) -> list[dict]:
        """
        Create index for FakeAVCeleb dataset.

        The function recursively searches for all .mp4 files in root_dir,
        infers label and fake type from each file path, and saves metadata
        into index json file.

        Args:
            index_path (str | Path): path where created index will be saved.

        Returns:
            index (list[dict]): list of dictionaries with metadata for each sample.
        """
        video_paths = sorted(self.root_dir.rglob("*.mp4"))

        if len(video_paths) == 0:
            raise RuntimeError(f"No .mp4 files found in {self.root_dir}")

        index = []

        print("Creating FakeAVCeleb Dataset Index")

        for idx, video_path in enumerate(tqdm(video_paths)):
            label, fake_type = infer_fakeavceleb_label(video_path)

            index.append(
                {
                    "sample_id": f"fakeavceleb_{idx:06d}",
                    "path": str(video_path),
                    "label": label,
                    "dataset": "fakeavceleb",
                    "fake_type": fake_type,
                }
            )

        write_json(index, str(index_path))

        return index
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from src.datasets.fakeavceleb import FakeAVCelebDataset
from src.utils.init_utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract WavLM and VideoMAE features for AV training."
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./data/indices/fakeavceleb",
    )
    parser.add_argument(
        "--audio-model-name-or-path",
        type=str,
        default="microsoft/wavlm-base-plus",
    )
    parser.add_argument(
        "--video-model-name-or-path",
        type=str,
        default="OpenGVLab/VideoMAEv2-Base",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--video-size", type=int, default=224)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild FakeAVCeleb train/val/test json indices.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop extraction on the first unreadable/corrupted sample instead of skipping it.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def load_backbones(audio_model_name_or_path: str, video_model_name_or_path: str, device: str):
    try:
        from transformers import (
            AutoConfig,
            AutoFeatureExtractor,
            AutoModel,
            VideoMAEImageProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "Feature extraction requires `transformers`. Install it with "
            "`pip install transformers`."
        ) from exc

    audio_processor = AutoFeatureExtractor.from_pretrained(audio_model_name_or_path)
    audio_model = AutoModel.from_pretrained(
        audio_model_name_or_path,
        low_cpu_mem_usage=False,
    ).to(device)
    audio_model.eval()

    video_config = AutoConfig.from_pretrained(
        video_model_name_or_path,
        trust_remote_code=True,
    )
    video_processor = VideoMAEImageProcessor.from_pretrained(video_model_name_or_path)
    video_model = AutoModel.from_pretrained(
        video_model_name_or_path,
        config=video_config,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    ).to(device)
    video_model.eval()

    return audio_processor, audio_model, video_processor, video_model


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return masked.sum(dim=1) / denom


def to_mono_resampled_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> np.ndarray:
    audio = audio.float()

    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    elif audio.ndim != 1:
        audio = audio.reshape(-1)

    audio_np = audio.detach().cpu().numpy().astype(np.float32, copy=False)

    if orig_sr <= 0:
        orig_sr = target_sr

    if orig_sr != target_sr and audio_np.size > 0:
        audio_np = resample_poly(audio_np, target_sr, orig_sr).astype(np.float32, copy=False)

    return audio_np


def extract_wavlm_feature(
    audio: torch.Tensor,
    audio_sample_rate: int,
    processor,
    model,
    device: str,
    target_sr: int,
) -> torch.Tensor:
    audio_np = to_mono_resampled_audio(audio, audio_sample_rate, target_sr)
    inputs = processor(
        audio_np,
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    feature = masked_mean(
        outputs.last_hidden_state,
        inputs.get("attention_mask"),
    )
    return feature.squeeze(0).cpu()


def extract_videomae_feature(
    video: torch.Tensor,
    processor,
    model,
    device: str,
) -> torch.Tensor:
    frames = [
        frame.detach().cpu().numpy().astype(np.float32, copy=False)
        for frame in video
    ]
    inputs = processor(frames, return_tensors="pt")

    if "pixel_values" not in inputs:
        raise KeyError("Video processor did not return `pixel_values`.")

    # VideoMAE-v2 HF model card expects [B, C, T, H, W].
    inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    feature = hidden_states.mean(dim=1)
    return feature.squeeze(0).cpu()


def build_dataset(args, split: str):
    return FakeAVCelebDataset(
        root_dir=args.dataset_root,
        index_dir=args.index_dir,
        name=split,
        num_frames=args.num_frames,
        video_size=args.video_size,
        limit=args.limit,
        rebuild_index=args.rebuild_index,
        instance_transforms=None,
    )


def save_split_features(output_dir: Path, audio_features, video_features, labels, meta):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(audio_features, output_dir / "audio_features.pt")
    torch.save(video_features, output_dir / "video_features.pt")
    torch.save(labels, output_dir / "labels.pt")
    torch.save(meta, output_dir / "meta.pt")


def save_skipped_samples(output_dir: Path, skipped_samples: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "skipped_samples.json").open("w", encoding="utf-8") as file_obj:
        json.dump(skipped_samples, file_obj, indent=2)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)

    (
        audio_processor,
        audio_model,
        video_processor,
        video_model,
    ) = load_backbones(
        audio_model_name_or_path=args.audio_model_name_or_path,
        video_model_name_or_path=args.video_model_name_or_path,
        device=device,
    )

    output_root = Path(args.output_root)

    print(f"Using device: {device}")
    print(f"Audio backbone: {args.audio_model_name_or_path}")
    print(f"Video backbone: {args.video_model_name_or_path}")

    for split in args.splits:
        dataset = build_dataset(args, split)
        audio_features = {}
        video_features = {}
        labels = {}
        meta = {}
        skipped_samples = []

        for sample_idx in tqdm(range(len(dataset)), desc=f"extract-{split}", total=len(dataset)):
            index_entry = dataset._index[sample_idx]

            try:
                sample = dataset[sample_idx]
                sample_id = sample["sample_id"]
                audio_features[sample_id] = extract_wavlm_feature(
                    audio=sample["audio"],
                    audio_sample_rate=int(sample.get("audio_sample_rate", 0)),
                    processor=audio_processor,
                    model=audio_model,
                    device=device,
                    target_sr=args.audio_target_sr,
                )
                video_features[sample_id] = extract_videomae_feature(
                    video=sample["video"],
                    processor=video_processor,
                    model=video_model,
                    device=device,
                )
                labels[sample_id] = int(sample["labels"].item())
                meta[sample_id] = {
                    "dataset": sample.get("dataset", "fakeavceleb"),
                    "fake_type": sample.get("fake_type", "unknown"),
                    "degradation": sample.get("degradation", "clean"),
                    "audio_degradation": sample.get("audio_degradation", "clean"),
                    "video_degradation": sample.get("video_degradation", "clean"),
                    "path": sample.get("path"),
                }
            except Exception as exc:
                if args.fail_on_error:
                    raise

                skipped_samples.append(
                    {
                        "sample_id": index_entry.get("sample_id"),
                        "path": index_entry.get("path"),
                        "error": str(exc),
                    }
                )
                print(
                    f"[{split}] skipped sample {index_entry.get('sample_id')} "
                    f"because of: {exc}"
                )

        save_split_features(
            output_dir=output_root / split,
            audio_features=audio_features,
            video_features=video_features,
            labels=labels,
            meta=meta,
        )
        save_skipped_samples(output_root / split, skipped_samples)

        first_audio_dim = next(iter(audio_features.values())).numel() if audio_features else 0
        first_video_dim = next(iter(video_features.values())).numel() if video_features else 0
        print(
            f"[{split}] saved {len(labels)} samples | skipped={len(skipped_samples)} | "
            f"audio_dim={first_audio_dim} | video_dim={first_video_dim}"
        )


if __name__ == "__main__":
    main()

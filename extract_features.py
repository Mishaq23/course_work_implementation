import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from src.datasets.degradations import (
    RawBoostConfig,
    add_audio_rawboost,
    add_audio_clipping,
    add_audio_impulsive_noise,
    add_audio_white_noise_snr,
    add_video_blur,
    add_video_downscale,
    add_video_gaussian_noise,
    add_video_low_light,
    add_video_quantization,
)
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
        "--audio-degradation",
        type=str,
        default="clean",
        choices=["clean", "white_noise", "clipping", "impulsive_noise", "rawboost"],
        help="Optional audio degradation applied before WavLM feature extraction.",
    )
    parser.add_argument(
        "--audio-white-noise-snr-db",
        type=float,
        default=20.0,
        help="SNR in dB for --audio-degradation white_noise.",
    )
    parser.add_argument(
        "--audio-clipping-threshold",
        type=float,
        default=0.6,
        help="Threshold for --audio-degradation clipping.",
    )
    parser.add_argument(
        "--audio-impulsive-max-percent",
        type=float,
        default=10.0,
        help="Max impulse percent for --audio-degradation impulsive_noise.",
    )
    parser.add_argument(
        "--audio-impulsive-gain",
        type=float,
        default=2.0,
        help="Impulse gain for --audio-degradation impulsive_noise.",
    )
    parser.add_argument(
        "--audio-rawboost-snr-min-db",
        type=float,
        default=10.0,
        help="RawBoost SSI minimum SNR in dB.",
    )
    parser.add_argument(
        "--audio-rawboost-snr-max-db",
        type=float,
        default=40.0,
        help="RawBoost SSI maximum SNR in dB.",
    )
    parser.add_argument(
        "--audio-rawboost-max-impulse-percent",
        type=float,
        default=10.0,
        help="RawBoost ISD max impulse percent.",
    )
    parser.add_argument(
        "--audio-rawboost-impulse-gain",
        type=float,
        default=2.0,
        help="RawBoost ISD impulse gain.",
    )
    parser.add_argument(
        "--video-degradation",
        type=str,
        default="clean",
        choices=[
            "clean",
            "downscale",
            "blur",
            "gaussian_noise",
            "low_light",
            "quantization",
        ],
        help="Optional video degradation applied before VideoMAE feature extraction.",
    )
    parser.add_argument(
        "--video-downscale-factor",
        type=float,
        default=0.5,
        help="Scale factor for --video-degradation downscale.",
    )
    parser.add_argument(
        "--video-blur-kernel-size",
        type=int,
        default=5,
        help="Odd kernel size for --video-degradation blur.",
    )
    parser.add_argument(
        "--video-gaussian-noise-std",
        type=float,
        default=0.05,
        help="Noise std for --video-degradation gaussian_noise.",
    )
    parser.add_argument(
        "--video-low-light-factor",
        type=float,
        default=0.5,
        help="Brightness factor for --video-degradation low_light.",
    )
    parser.add_argument(
        "--video-quantization-levels",
        type=int,
        default=32,
        help="Quantization levels for --video-degradation quantization.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild FakeAVCeleb train/val/test json indices.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="id_component",
        choices=["sample", "id_tuple", "id_component"],
        help="How to split FakeAVCeleb into train/val/test partitions.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Skip the safety check that aborts extraction when train/val/test ids overlap.",
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


def pool_audio_hidden_states(
    model,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Pool WavLM hidden states with the correct feature-level attention mask.

    WavLM receives an attention mask in waveform time steps, while
    `last_hidden_state` is already temporally downsampled. We therefore need to
    convert the input mask to feature-vector resolution before masked pooling.
    """
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    if hasattr(model, "_get_feature_vector_attention_mask"):
        feature_attention_mask = model._get_feature_vector_attention_mask(
            hidden_states.shape[1],
            attention_mask,
        )
        return masked_mean(hidden_states, feature_attention_mask)

    return hidden_states.mean(dim=1)


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

    feature = pool_audio_hidden_states(
        model,
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
    inputs = processor([frames], return_tensors="pt")

    if "pixel_values" not in inputs:
        raise KeyError("Video processor did not return `pixel_values`.")

    pixel_values = inputs["pixel_values"]

    # Be tolerant to processor variants and normalize to [B, C, T, H, W].
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    elif pixel_values.ndim != 5:
        raise ValueError(
            "Expected video processor output with 4 or 5 dims, "
            f"got shape {tuple(pixel_values.shape)}."
        )

    # Most processors return [B, T, C, H, W], while VideoMAE models expect [B, C, T, H, W].
    if pixel_values.shape[1] != 3 and pixel_values.shape[2] == 3:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

    inputs["pixel_values"] = pixel_values
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feature = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        feature = outputs.last_hidden_state
    else:
        feature = outputs[0]

    if feature.ndim == 1:
        return feature.cpu()

    if feature.ndim == 2:
        return feature.squeeze(0).cpu() if feature.shape[0] == 1 else feature.mean(dim=0).cpu()

    reduce_dims = tuple(range(1, feature.ndim - 1))
    pooled = feature.mean(dim=reduce_dims)
    return pooled.squeeze(0).cpu()


def build_dataset(args, split: str):
    return FakeAVCelebDataset(
        root_dir=args.dataset_root,
        index_dir=args.index_dir,
        name=split,
        num_frames=args.num_frames,
        video_size=args.video_size,
        limit=args.limit,
        split_strategy=args.split_strategy,
        rebuild_index=args.rebuild_index,
        instance_transforms=None,
    )


def apply_audio_degradation(
    audio: torch.Tensor,
    audio_sample_rate: int,
    args,
    sample_idx: int,
) -> tuple[torch.Tensor, str]:
    if args.audio_degradation == "clean":
        return audio, "clean"

    if args.audio_degradation == "white_noise":
        degraded = add_audio_white_noise_snr(
            audio=audio,
            snr_db=args.audio_white_noise_snr_db,
            seed=args.seed + sample_idx,
        )
        degradation_name = f"white_noise_snr_{args.audio_white_noise_snr_db:g}db"
        return degraded, degradation_name

    if args.audio_degradation == "clipping":
        degraded = add_audio_clipping(
            audio=audio,
            clipping_threshold=args.audio_clipping_threshold,
        )
        degradation_name = f"clipping_t{args.audio_clipping_threshold:g}"
        return degraded, degradation_name

    if args.audio_degradation == "impulsive_noise":
        degraded = add_audio_impulsive_noise(
            audio=audio,
            max_percent=args.audio_impulsive_max_percent,
            gain=args.audio_impulsive_gain,
            seed=args.seed + sample_idx,
        )
        degradation_name = (
            "impulsive_noise"
            f"_p{args.audio_impulsive_max_percent:g}"
            f"_g{args.audio_impulsive_gain:g}"
        )
        return degraded, degradation_name

    if args.audio_degradation == "rawboost":
        sample_rate = int(audio_sample_rate) if int(audio_sample_rate) > 0 else args.audio_target_sr
        config = RawBoostConfig(
            sample_rate=sample_rate,
            max_impulse_percent=args.audio_rawboost_max_impulse_percent,
            impulse_gain=args.audio_rawboost_impulse_gain,
            snr_min_db=args.audio_rawboost_snr_min_db,
            snr_max_db=args.audio_rawboost_snr_max_db,
        )
        degraded = add_audio_rawboost(
            audio=audio,
            config=config,
            seed=args.seed + sample_idx,
        )
        degradation_name = (
            "rawboost"
            f"_snr{args.audio_rawboost_snr_min_db:g}-{args.audio_rawboost_snr_max_db:g}"
            f"_imp{args.audio_rawboost_max_impulse_percent:g}"
        )
        return degraded, degradation_name

    raise ValueError(f"Unsupported audio degradation: {args.audio_degradation}")


def apply_video_degradation(
    video: torch.Tensor,
    args,
    sample_idx: int,
) -> tuple[torch.Tensor, str]:
    if args.video_degradation == "clean":
        return video, "clean"

    if args.video_degradation == "downscale":
        degraded = add_video_downscale(
            video=video,
            scale_factor=args.video_downscale_factor,
        )
        degradation_name = f"downscale_x{args.video_downscale_factor:g}"
        return degraded, degradation_name

    if args.video_degradation == "blur":
        degraded = add_video_blur(
            video=video,
            kernel_size=args.video_blur_kernel_size,
        )
        degradation_name = f"blur_k{args.video_blur_kernel_size}"
        return degraded, degradation_name

    if args.video_degradation == "gaussian_noise":
        degraded = add_video_gaussian_noise(
            video=video,
            std=args.video_gaussian_noise_std,
            seed=args.seed + sample_idx,
        )
        degradation_name = f"gaussian_noise_std{args.video_gaussian_noise_std:g}"
        return degraded, degradation_name

    if args.video_degradation == "low_light":
        degraded = add_video_low_light(
            video=video,
            factor=args.video_low_light_factor,
        )
        degradation_name = f"low_light_f{args.video_low_light_factor:g}"
        return degraded, degradation_name

    if args.video_degradation == "quantization":
        degraded = add_video_quantization(
            video=video,
            levels=args.video_quantization_levels,
        )
        degradation_name = f"quantization_l{args.video_quantization_levels}"
        return degraded, degradation_name

    raise ValueError(f"Unsupported video degradation: {args.video_degradation}")


def build_overall_degradation_label(
    audio_degradation: str,
    video_degradation: str,
) -> str:
    if audio_degradation == "clean" and video_degradation == "clean":
        return "clean"
    return f"audio:{audio_degradation}|video:{video_degradation}"


def extract_ids_from_index(index: list[dict]) -> set[str]:
    id_pattern = re.compile(r"id\d+")
    ids = set()
    for item in index:
        source = str(item.get("relative_path", item.get("path", ""))).lower()
        ids.update(id_pattern.findall(source))
    return ids


def compute_split_summary(datasets: dict[str, FakeAVCelebDataset], split_strategy: str) -> dict:
    summary = {
        "split_strategy": split_strategy,
        "splits": {},
        "overlap_by_id": {},
    }

    split_ids: dict[str, set[str]] = {}
    split_names = list(datasets.keys())

    for split_name, dataset in datasets.items():
        labels = [int(item["label"]) for item in dataset._index]
        num_samples = len(labels)
        num_fake = sum(labels)
        split_ids[split_name] = extract_ids_from_index(dataset._index)
        summary["splits"][split_name] = {
            "num_samples": num_samples,
            "num_fake": num_fake,
            "num_real": num_samples - num_fake,
            "fake_ratio": (num_fake / num_samples) if num_samples else None,
        }

    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            summary["overlap_by_id"][f"{left}__{right}"] = {
                "count": len(overlap),
                "ids": sorted(overlap),
            }

    return summary


def validate_split_summary(summary: dict, allow_overlap: bool) -> None:
    overlap_counts = {
        pair: info["count"]
        for pair, info in summary["overlap_by_id"].items()
    }
    max_overlap = max(overlap_counts.values(), default=0)

    print("Split summary:")
    for split_name, stats in summary["splits"].items():
        print(
            f"  {split_name}: n={stats['num_samples']} "
            f"fake={stats['num_fake']} real={stats['num_real']} "
            f"fake_ratio={stats['fake_ratio']:.4f}"
        )
    for pair, count in overlap_counts.items():
        print(f"  overlap[{pair}]={count}")

    if max_overlap > 0 and not allow_overlap and summary["split_strategy"] != "sample":
        raise RuntimeError(
            "Detected identity overlap between splits before extraction. "
            "Refusing to continue because this would produce leaky features. "
            "If you intentionally want sample-level overlap, rerun with "
            "`--split-strategy sample --allow-overlap`."
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
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Audio backbone: {args.audio_model_name_or_path}")
    print(f"Video backbone: {args.video_model_name_or_path}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Audio degradation: {args.audio_degradation}")
    print(f"Video degradation: {args.video_degradation}")

    datasets = {split: build_dataset(args, split) for split in args.splits}
    split_summary = compute_split_summary(datasets, args.split_strategy)
    with (output_root / "split_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(split_summary, file_obj, indent=2)
    validate_split_summary(split_summary, allow_overlap=args.allow_overlap)

    for split in args.splits:
        dataset = datasets[split]
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
                degraded_audio, audio_degradation_name = apply_audio_degradation(
                    audio=sample["audio"],
                    audio_sample_rate=int(sample.get("audio_sample_rate", 0)),
                    args=args,
                    sample_idx=sample_idx,
                )
                degraded_video, video_degradation_name = apply_video_degradation(
                    video=sample["video"],
                    args=args,
                    sample_idx=sample_idx,
                )
                audio_feature = extract_wavlm_feature(
                    audio=degraded_audio,
                    audio_sample_rate=int(sample.get("audio_sample_rate", 0)),
                    processor=audio_processor,
                    model=audio_model,
                    device=device,
                    target_sr=args.audio_target_sr,
                )
                video_feature = extract_videomae_feature(
                    video=degraded_video,
                    processor=video_processor,
                    model=video_model,
                    device=device,
                )

                audio_features[sample_id] = audio_feature
                video_features[sample_id] = video_feature
                labels[sample_id] = int(sample["labels"].item())
                meta[sample_id] = {
                    "dataset": sample.get("dataset", "fakeavceleb"),
                    "fake_type": sample.get("fake_type", "unknown"),
                    "degradation": build_overall_degradation_label(
                        audio_degradation=audio_degradation_name,
                        video_degradation=video_degradation_name,
                    ),
                    "audio_degradation": audio_degradation_name,
                    "video_degradation": video_degradation_name,
                    "path": sample.get("path"),
                    "relative_path": index_entry.get("relative_path"),
                    "split_group_key": index_entry.get("split_group_key", index_entry.get("group_key")),
                    "split_strategy": args.split_strategy,
                    "split_name": split,
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

# Audio-Visual Deepfake Detection

PyTorch/Hydra codebase for binary deepfake detection with audio-only, video-only, and fused audio-video models.

The project supports two training modes:

- raw audio/video training from datasets like `FakeAVCeleb`
- feature-level training from precomputed embeddings such as WavLM / VideoMAE outputs

## Project structure

```text
train.py                     # training entry point
inference.py                 # evaluation / prediction entry point
src/
  configs/                   # Hydra configs
  datasets/                  # raw datasets, feature datasets, collate, degradations
  model/                     # audio/video encoders, fusion blocks, classifiers
  trainer/                   # training and inference loops
  metrics/                   # binary classification metrics
  loss/                      # BCEWithLogits loss
  logger/                    # dummy, WandB, Comet writers
  transforms/                # optional batch / instance transforms
tests/                       # lightweight unit tests on synthetic data
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layouts

### Raw video dataset

For `FakeAVCelebDataset`, point `dataset_root` to the directory containing `.mp4` files. Labels are inferred from path fragments such as `RealVideo/RealAudio`, `FakeVideo/RealAudio`, `RealVideo/FakeAudio`, `FakeVideo/FakeAudio`.

Example:

```text
/path/to/FakeAVCeleb/
  RealVideo-RealAudio/...
  FakeVideo-RealAudio/...
  RealVideo-FakeAudio/...
  FakeVideo-FakeAudio/...
```

`FakeAVCelebDataset` now uses a group-wise split by default (`split_strategy=id_component`),
so train/val/test are separated by connected identity components extracted from the
path rather than by individual samples. If you previously extracted features with an
older split, regenerate them before training new models.

### Feature dataset

Point `features_root` to a directory with split subfolders:

```text
features_root/
  train/
    audio_features.pt
    video_features.pt
    labels.pt
    meta.pt
  val/
    ...
  test/
    ...
```

Each `.pt` file is expected to be a dictionary keyed by `sample_id`.

## Main configs

- `src/configs/baseline.yaml`: default raw AV training
- `src/configs/audio_pretrain.yaml`: audio-only feature training
- `src/configs/video_pretrain.yaml`: video-only feature training
- `src/configs/av_finetune.yaml`: fused AV feature training from pretrained audio/video encoders
- `src/configs/inference.yaml`: evaluation / prediction config

## Training

### 1. Raw audio-video training

```bash
.venv/bin/python train.py \
  dataset_root=/path/to/FakeAVCeleb \
  writer=dummy \
  trainer.override=True
```

### 2. Audio-only feature pretraining

```bash
.venv/bin/python train.py \
  -cn=audio_pretrain \
  features_root=/path/to/features \
  writer=dummy \
  trainer.override=True
```

### 3. Video-only feature pretraining

```bash
.venv/bin/python train.py \
  -cn=video_pretrain \
  features_root=/path/to/features \
  writer=dummy \
  trainer.override=True
```

### 4. Final AV fine-tuning from pretrained branches

By default, `av_finetune.yaml` looks for:

- `saved/audio_pretrain/model_best.pth`
- `saved/video_pretrain/model_best.pth`

Run:

```bash
.venv/bin/python train.py \
  -cn=av_finetune \
  features_root=/path/to/features \
  writer=dummy \
  trainer.override=True
```

If your checkpoint names differ:

```bash
.venv/bin/python train.py \
  -cn=av_finetune \
  features_root=/path/to/features \
  writer=dummy \
  trainer.override=True \
  trainer.from_pretrained.0.path=saved/YOUR_AUDIO_RUN/model_best.pth \
  trainer.from_pretrained.1.path=saved/YOUR_VIDEO_RUN/model_best.pth
```

### Freezing encoders

To train only fusion/head layers:

```bash
.venv/bin/python train.py \
  -cn=av_finetune \
  features_root=/path/to/features \
  writer=dummy \
  trainer.freeze_modules='[audio_encoder,video_encoder]' \
  trainer.override=True
```

## Evaluation / inference

Evaluate a trained checkpoint on feature splits:

```bash
.venv/bin/python inference.py \
  features_root=/path/to/features \
  inferencer.from_pretrained=saved/av_finetune/model_best.pth \
  inferencer.save_path=saved/predictions
```

For modality-specific feature models, use matching dataset labels:

```bash
.venv/bin/python inference.py \
  model=audio_only_model \
  datasets=features_audio \
  features_root=/path/to/features \
  inferencer.device_tensors='[audio,labels]' \
  inferencer.from_pretrained=saved/audio_pretrain/model_best.pth \
  inferencer.save_path=saved/audio_predictions
```

```bash
.venv/bin/python inference.py \
  model=video_only_model \
  datasets=features_video \
  features_root=/path/to/features \
  inferencer.device_tensors='[video,labels]' \
  inferencer.from_pretrained=saved/video_pretrain/model_best.pth \
  inferencer.save_path=saved/video_predictions
```

For raw-data evaluation, override dataset/model choices:

```bash
.venv/bin/python inference.py \
  datasets=fakeavceleb \
  model=av_baseline_model \
  dataset_root=/path/to/FakeAVCeleb \
  inferencer.from_pretrained=saved/debug/model_best.pth \
  inferencer.save_path=saved/raw_predictions
```


## Metrics

Binary metrics are computed from logits as follows:

- `BCEWithLogitsLoss` receives raw logits
- `Accuracy`, `Precision`, `Recall`, `F1` use sigmoid + threshold
- `AUROC` and `PRAUC` use probabilities over the full epoch, not per-batch averages

If an evaluation partition contains only one class, ranking metrics such as `AUROC` and `PRAUC` return `NaN`.

## Kaggle example

Raw-data training:

```bash
!python train.py \
  dataset_root=/kaggle/input/fakeavceleb/FakeAVCeleb \
  writer=dummy \
  trainer.device=cuda \
  trainer.override=True
```

Feature-level AV fine-tuning:

```bash
!python train.py \
  -cn=av_finetune \
  features_root=/kaggle/input/av-features \
  writer=dummy \
  trainer.device=cuda \
  trainer.override=True
```

## Notes

- Default logging uses `writer=dummy` so the project runs without authentication.
- `writer=wandb` and `writer=cometml` remain available as optional overrides.
- Raw video loading uses `torchvision.io.read_video`, so the environment must have working FFmpeg support.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal


def _get_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _uniform(
    low: float,
    high: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> float:
    value = torch.empty(1, device=device).uniform_(
        low,
        high,
        generator=generator,
    )
    return float(value.item())


def normalize_waveform(
    audio: torch.Tensor,
    always: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    max_abs = audio.abs().amax().clamp_min(eps)

    if always or max_abs > 1.0:
        return audio / max_abs

    return audio


def _as_channels_first_audio(audio: torch.Tensor) -> tuple[torch.Tensor, bool]:
    squeeze_back = audio.ndim == 1

    if squeeze_back:
        audio = audio.unsqueeze(0)

    if audio.ndim != 2:
        raise ValueError(
            "Audio degradation expects waveform with shape [L] or [C, L]. "
            f"Got {tuple(audio.shape)}."
        )

    return audio.float(), squeeze_back


def _to_numpy_audio(audio: torch.Tensor) -> tuple[np.ndarray, bool, torch.device, torch.dtype]:
    audio, squeeze_back = _as_channels_first_audio(audio)
    return audio.detach().cpu().numpy(), squeeze_back, audio.device, audio.dtype


def _from_numpy_audio(
    audio: np.ndarray,
    squeeze_back: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).to(
        device=device,
        dtype=dtype,
    )
    return tensor.squeeze(0) if squeeze_back else tensor


def _get_numpy_rng(seed: int | None):
    if seed is None:
        return np.random

    return np.random.RandomState(seed)


def _np_rand_range(
    rng,
    low: float,
    high: float,
    integer: bool,
) -> float | int:
    value = rng.uniform(low=low, high=high, size=(1,))

    if integer:
        return int(value[0])

    return float(value[0])


def _np_normalize_waveform(
    audio: np.ndarray,
    always: bool,
    eps: float = 1e-8,
) -> np.ndarray:
    max_abs = np.maximum(np.amax(np.abs(audio)), eps)

    if always or max_abs > 1.0:
        return audio / max_abs

    return audio


def _gen_notch_coeffs(
    rng,
    n_bands: int,
    min_freq: float,
    max_freq: float,
    min_bandwidth: float,
    max_bandwidth: float,
    min_coeff: int,
    max_coeff: int,
    min_gain_db: float,
    max_gain_db: float,
    sample_rate: int,
) -> np.ndarray:
    b = np.array([1.0], dtype=np.float64)

    for _ in range(n_bands):
        center_freq = _np_rand_range(rng, min_freq, max_freq, integer=False)
        bandwidth = _np_rand_range(
            rng,
            min_bandwidth,
            max_bandwidth,
            integer=False,
        )
        coeff_count = _np_rand_range(rng, min_coeff, max_coeff, integer=True)

        if coeff_count % 2 == 0:
            coeff_count += 1

        low_freq = center_freq - bandwidth / 2.0
        high_freq = center_freq + bandwidth / 2.0

        if low_freq <= 0:
            low_freq = 1.0 / 1000.0

        if high_freq >= sample_rate / 2.0:
            high_freq = sample_rate / 2.0 - 1.0 / 1000.0

        notch = signal.firwin(
            coeff_count,
            [float(low_freq), float(high_freq)],
            window="hamming",
            fs=sample_rate,
        )
        b = np.convolve(notch, b)

        gain_db = _np_rand_range(rng, min_gain_db, max_gain_db, integer=False)
        _, response = signal.freqz(b, 1, fs=sample_rate)
        b = (10.0 ** (gain_db / 20.0)) * b / np.amax(np.abs(response))

    return b


def _filter_fir(audio: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    pad_size = coeffs.shape[0] + 1
    padded = np.pad(audio, (0, pad_size), "constant")
    filtered = signal.lfilter(coeffs, 1, padded)
    return filtered[int(pad_size / 2) : int(filtered.shape[0] - pad_size / 2)]


def _rawboost_lnl_1d(
    audio: np.ndarray,
    rng,
    sample_rate: int,
    n_filter_orders: int,
    n_bands: int,
    min_freq: float,
    max_freq: float,
    min_bandwidth: float,
    max_bandwidth: float,
    min_coeff: int,
    max_coeff: int,
    min_gain_db: float,
    max_gain_db: float,
    min_bias_lnl: float,
    max_bias_lnl: float,
) -> np.ndarray:
    degraded = np.zeros_like(audio, dtype=np.float64)

    for order_index in range(n_filter_orders):
        if order_index == 1:
            min_gain_db = min_gain_db - min_bias_lnl
            max_gain_db = max_gain_db - max_bias_lnl

        coeffs = _gen_notch_coeffs(
            rng=rng,
            n_bands=n_bands,
            min_freq=min_freq,
            max_freq=max_freq,
            min_bandwidth=min_bandwidth,
            max_bandwidth=max_bandwidth,
            min_coeff=min_coeff,
            max_coeff=max_coeff,
            min_gain_db=min_gain_db,
            max_gain_db=max_gain_db,
            sample_rate=sample_rate,
        )
        degraded = degraded + _filter_fir(
            np.power(audio, order_index + 1),
            coeffs,
        )

    degraded = degraded - np.mean(degraded)
    return _np_normalize_waveform(degraded, always=False)


def _rawboost_isd_1d(audio: np.ndarray, rng, max_percent: float, gain: float) -> np.ndarray:
    beta = _np_rand_range(rng, 0.0, max_percent, integer=False)
    degraded = np.copy(audio)
    n_samples = audio.shape[0]
    n_impulses = int(n_samples * (beta / 100.0))

    if n_impulses == 0:
        return degraded

    positions = rng.permutation(n_samples)[:n_impulses]
    random_factors = (
        (2.0 * rng.rand(positions.shape[0]) - 1.0)
        * (2.0 * rng.rand(positions.shape[0]) - 1.0)
    )
    impulses = gain * audio[positions] * random_factors
    degraded[positions] = audio[positions] + impulses
    return _np_normalize_waveform(degraded, always=False)


def _rawboost_ssi_1d(
    audio: np.ndarray,
    rng,
    sample_rate: int,
    snr_min_db: float,
    snr_max_db: float,
    n_bands: int,
    min_freq: float,
    max_freq: float,
    min_bandwidth: float,
    max_bandwidth: float,
    min_coeff: int,
    max_coeff: int,
    min_gain_db: float,
    max_gain_db: float,
) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, audio.shape[0])
    coeffs = _gen_notch_coeffs(
        rng=rng,
        n_bands=n_bands,
        min_freq=min_freq,
        max_freq=max_freq,
        min_bandwidth=min_bandwidth,
        max_bandwidth=max_bandwidth,
        min_coeff=min_coeff,
        max_coeff=max_coeff,
        min_gain_db=min_gain_db,
        max_gain_db=max_gain_db,
        sample_rate=sample_rate,
    )
    noise = _filter_fir(noise, coeffs)
    noise = _np_normalize_waveform(noise, always=True)

    snr_db = _np_rand_range(rng, snr_min_db, snr_max_db, integer=False)
    noise = (
        noise
        / max(np.linalg.norm(noise, 2), 1e-8)
        * max(np.linalg.norm(audio, 2), 1e-8)
        / (10.0 ** (0.05 * snr_db))
    )
    return audio + noise


def add_audio_white_noise_snr(
    audio: torch.Tensor,
    snr_db: float,
    seed: int | None = None,
) -> torch.Tensor:
    audio, squeeze_back = _as_channels_first_audio(audio)
    generator = _get_generator(seed, audio.device)

    noise = torch.randn(
        audio.shape,
        generator=generator,
        device=audio.device,
        dtype=audio.dtype,
    )
    signal_norm = audio.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    noise = noise / noise_norm * signal_norm / (10.0 ** (snr_db / 20.0))

    degraded = normalize_waveform(audio + noise)
    return degraded.squeeze(0) if squeeze_back else degraded


def add_audio_clipping(audio: torch.Tensor, clipping_threshold: float = 0.6) -> torch.Tensor:
    degraded = audio.float().clamp(
        min=-clipping_threshold,
        max=clipping_threshold,
    )
    return normalize_waveform(degraded)


def add_audio_impulsive_noise(
    audio: torch.Tensor,
    max_percent: float = 10.0,
    gain: float = 2.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    RawBoost impulsive signal-dependent additive noise.
    """
    audio_np, squeeze_back, device, dtype = _to_numpy_audio(audio)
    rng = _get_numpy_rng(seed)
    degraded = np.stack(
        [
            _rawboost_isd_1d(
                channel,
                rng=rng,
                max_percent=max_percent,
                gain=gain,
            )
            for channel in audio_np
        ],
        axis=0,
    )
    return _from_numpy_audio(degraded, squeeze_back, device, dtype)


def add_audio_lnl_convolutive_noise(
    audio: torch.Tensor,
    sample_rate: int,
    n_filter_orders: int = 5,
    n_bands: int = 5,
    min_freq: float = 20.0,
    max_freq: float = 8000.0,
    min_bandwidth: float = 100.0,
    max_bandwidth: float = 1000.0,
    min_coeff: int = 10,
    max_coeff: int = 100,
    min_gain_db: float = 0.0,
    max_gain_db: float = 0.0,
    min_bias_lnl: float = 20.0,
    max_bias_lnl: float = 5.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    RawBoost linear and non-linear convolutive noise.

    This follows the official RawBoost FIR-notch implementation.
    """
    audio_np, squeeze_back, device, dtype = _to_numpy_audio(audio)
    rng = _get_numpy_rng(seed)
    max_freq = min(max_freq, sample_rate / 2.0 - 1e-3)

    degraded = np.stack(
        [
            _rawboost_lnl_1d(
                channel,
                rng=rng,
                sample_rate=sample_rate,
                n_filter_orders=n_filter_orders,
                n_bands=n_bands,
                min_freq=min_freq,
                max_freq=max_freq,
                min_bandwidth=min_bandwidth,
                max_bandwidth=max_bandwidth,
                min_coeff=min_coeff,
                max_coeff=max_coeff,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                min_bias_lnl=min_bias_lnl,
                max_bias_lnl=max_bias_lnl,
            )
            for channel in audio_np
        ],
        axis=0,
    )
    return _from_numpy_audio(degraded, squeeze_back, device, dtype)


def add_audio_filtered_noise_snr(
    audio: torch.Tensor,
    sample_rate: int,
    snr_min_db: float = 10.0,
    snr_max_db: float = 40.0,
    n_bands: int = 5,
    min_freq: float = 20.0,
    max_freq: float = 8000.0,
    min_bandwidth: float = 100.0,
    max_bandwidth: float = 1000.0,
    min_coeff: int = 10,
    max_coeff: int = 100,
    min_gain_db: float = 0.0,
    max_gain_db: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    RawBoost stationary signal-independent additive noise.
    """
    audio_np, squeeze_back, device, dtype = _to_numpy_audio(audio)
    rng = _get_numpy_rng(seed)
    max_freq = min(max_freq, sample_rate / 2.0 - 1e-3)

    degraded = np.stack(
        [
            _rawboost_ssi_1d(
                channel,
                rng=rng,
                sample_rate=sample_rate,
                snr_min_db=snr_min_db,
                snr_max_db=snr_max_db,
                n_bands=n_bands,
                min_freq=min_freq,
                max_freq=max_freq,
                min_bandwidth=min_bandwidth,
                max_bandwidth=max_bandwidth,
                min_coeff=min_coeff,
                max_coeff=max_coeff,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
            )
            for channel in audio_np
        ],
        axis=0,
    )
    return _from_numpy_audio(degraded, squeeze_back, device, dtype)


def add_audio_rawboost(
    audio: torch.Tensor,
    config: RawBoostConfig,
    seed: int | None = None,
) -> torch.Tensor:
    degraded = audio.float()

    if config.apply_lnl:
        degraded = add_audio_lnl_convolutive_noise(
            degraded,
            sample_rate=config.sample_rate,
            n_filter_orders=config.n_filter_orders,
            n_bands=config.n_bands,
            min_freq=config.min_freq,
            max_freq=config.max_freq,
            min_bandwidth=config.min_bandwidth,
            max_bandwidth=config.max_bandwidth,
            min_coeff=config.min_coeff,
            max_coeff=config.max_coeff,
            min_gain_db=config.lnl_min_gain_db,
            max_gain_db=config.lnl_max_gain_db,
            min_bias_lnl=config.lnl_min_bias_db,
            max_bias_lnl=config.lnl_max_bias_db,
            seed=seed,
        )

    if config.apply_isd:
        degraded = add_audio_impulsive_noise(
            degraded,
            max_percent=config.max_impulse_percent,
            gain=config.impulse_gain,
            seed=None if seed is None else seed + 1,
        )

    if config.apply_ssi:
        degraded = add_audio_filtered_noise_snr(
            degraded,
            sample_rate=config.sample_rate,
            snr_min_db=config.snr_min_db,
            snr_max_db=config.snr_max_db,
            n_bands=config.n_bands,
            min_freq=config.min_freq,
            max_freq=config.max_freq,
            min_bandwidth=config.min_bandwidth,
            max_bandwidth=config.max_bandwidth,
            min_coeff=config.min_coeff,
            max_coeff=config.max_coeff,
            min_gain_db=config.ssi_min_gain_db,
            max_gain_db=config.ssi_max_gain_db,
            seed=None if seed is None else seed + 2,
        )

    return normalize_waveform(degraded)


@dataclass
class RawBoostConfig:
    sample_rate: int
    apply_lnl: bool = True
    apply_isd: bool = True
    apply_ssi: bool = True
    n_filter_orders: int = 5
    n_bands: int = 5
    min_freq: float = 20.0
    max_freq: float = 8000.0
    min_bandwidth: float = 100.0
    max_bandwidth: float = 1000.0
    min_coeff: int = 10
    max_coeff: int = 100
    lnl_min_gain_db: float = 0.0
    lnl_max_gain_db: float = 0.0
    lnl_min_bias_db: float = 20.0
    lnl_max_bias_db: float = 5.0
    max_impulse_percent: float = 10.0
    impulse_gain: float = 2.0
    snr_min_db: float = 10.0
    snr_max_db: float = 40.0
    ssi_min_gain_db: float = 0.0
    ssi_max_gain_db: float = 0.0


class RawBoostDegradation:
    """
    Composable RawBoost-style waveform degradation.

    This follows the three RawBoost families: linear/non-linear convolutive
    distortion, impulsive signal-dependent noise, and stationary independent
    noise. It is intended for stress testing.
    """

    def __init__(self, config: RawBoostConfig):
        self.config = config

    def __call__(self, audio: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        return add_audio_rawboost(audio, config=self.config, seed=seed)


def add_video_gaussian_noise(
    video: torch.Tensor,
    std: float = 0.05,
    seed: int | None = None,
) -> torch.Tensor:
    generator = _get_generator(seed, video.device)
    noise = torch.randn(
        video.shape,
        generator=generator,
        device=video.device,
        dtype=video.dtype,
    )
    return (video.float() + noise * std).clamp(0.0, 1.0)


def add_video_low_light(video: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    return (video.float() * factor).clamp(0.0, 1.0)


def add_video_blur(video: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    squeeze_back = video.ndim == 4

    if squeeze_back:
        video = video.unsqueeze(0)

    if video.ndim != 5:
        raise ValueError(
            "Video degradation expects [T, C, H, W] or [B, T, C, H, W]. "
            f"Got {tuple(video.shape)}."
        )

    b, t, c, h, w = video.shape
    flat_video = video.float().reshape(b * t, c, h, w)
    kernel = torch.ones(
        c,
        1,
        kernel_size,
        kernel_size,
        device=video.device,
        dtype=video.dtype,
    )
    kernel = kernel / float(kernel_size * kernel_size)
    padding = kernel_size // 2

    blurred = F.conv2d(
        flat_video,
        kernel,
        padding=padding,
        groups=c,
    )
    blurred = blurred.reshape(b, t, c, h, w).clamp(0.0, 1.0)
    return blurred.squeeze(0) if squeeze_back else blurred


def add_video_downscale(video: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    squeeze_back = video.ndim == 4

    if squeeze_back:
        video = video.unsqueeze(0)

    if video.ndim != 5:
        raise ValueError(
            "Video degradation expects [T, C, H, W] or [B, T, C, H, W]. "
            f"Got {tuple(video.shape)}."
        )

    b, t, c, h, w = video.shape
    flat_video = video.float().reshape(b * t, c, h, w)
    small = F.interpolate(
        flat_video,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    restored = F.interpolate(
        small,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    restored = restored.reshape(b, t, c, h, w).clamp(0.0, 1.0)
    return restored.squeeze(0) if squeeze_back else restored


def add_video_quantization(video: torch.Tensor, levels: int = 32) -> torch.Tensor:
    if levels < 2:
        raise ValueError("levels must be >= 2.")

    video = video.float().clamp(0.0, 1.0)
    return torch.round(video * (levels - 1)) / float(levels - 1)


AUDIO_DEGRADATIONS = {
    "white_noise": add_audio_white_noise_snr,
    "clipping": add_audio_clipping,
    "impulsive_noise": add_audio_impulsive_noise,
    "filtered_noise": add_audio_filtered_noise_snr,
    "lnl_convolutive": add_audio_lnl_convolutive_noise,
}

VIDEO_DEGRADATIONS = {
    "gaussian_noise": add_video_gaussian_noise,
    "low_light": add_video_low_light,
    "blur": add_video_blur,
    "downscale": add_video_downscale,
    "quantization": add_video_quantization,
}

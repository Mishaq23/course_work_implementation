from pathlib import Path


def infer_label_from_path(path: str | Path) -> tuple[int, str]:
    path_str = str(path).lower()

    real_video = "RealVideo" in path_str
    fake_video = "FakeVideo" in path_str
    real_audio = "RealAudio" in path_str
    fake_audio = "FakeAudio" in path_str

    if real_video and real_audio:
        return 0, "real"

    if fake_video and real_audio:
        return 1, "video_fake"

    if real_video and fake_audio:
        return 1, "audio_fake"

    if fake_video and fake_audio:
        return 1, "av_fake"

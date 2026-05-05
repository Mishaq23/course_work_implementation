from pathlib import Path


def infer_fakeavceleb_label(path: str | Path) -> tuple[int, str]:
    path_str = str(path).lower()

    real_video = "realvideo" in path_str
    fake_video = "fakevideo" in path_str
    real_audio = "realaudio" in path_str
    fake_audio = "fakeaudio" in path_str

    if real_video and real_audio:
        return 0, "real"

    if fake_video and real_audio:
        return 1, "video_fake"

    if real_video and fake_audio:
        return 1, "audio_fake"

    if fake_video and fake_audio:
        return 1, "av_fake"

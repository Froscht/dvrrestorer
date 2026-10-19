import cv2
import numpy as np


def test_make_clean_clip_writes_readable_mp4(tmp_path):
    from scripts.synthesize_dvr_artifacts import make_clean_clip
    out = tmp_path / "clean.mp4"
    make_clean_clip(out, duration_s=1.0, fps=30.0, size=(320, 180))
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30


def test_add_chroma_noise_increases_chroma_std():
    from scripts.synthesize_dvr_artifacts import add_chroma_noise
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    yuv_before = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    noisy = add_chroma_noise(frame, strength=0.5)
    yuv_after = cv2.cvtColor(noisy, cv2.COLOR_BGR2YUV)
    assert yuv_after[:, :, 1].std() > yuv_before[:, :, 1].std() + 5

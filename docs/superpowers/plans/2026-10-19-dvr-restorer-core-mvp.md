# DVR Restorer Core + CLI (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a headless CLI that takes an analog FPV DVR clip and writes a cleaned MP4 with (a) low-light boost, (b) RF-stripe removal, (c) chroma-noise denoise — all OSD-aware — so the pipeline works end-to-end before the GUI is built on top of it.

**Architecture:** Pure-function `core/` modules (`scan`, `osd`, `fix_dark`, `fix_lines`, `fix_chroma`, `pipeline`, `video_io`) wired through a `pipeline.run` entry; a Timeline is just a list of per-track keyframes `(t_sec, strength)`; `profiles.py` persists it as JSON; `cli.py` is a thin ArgumentParser over `pipeline.run`.

**Tech Stack:** Python 3.11+, OpenCV 4.x, numpy, scipy, onnxruntime (CPU default), imageio-ffmpeg (ffmpeg binary), pytest. No GUI libs in this plan.

**Spec:** `docs/superpowers/specs/2026-10-19-dvr-restorer-design.md`

## Global Constraints

- Python 3.11+ (uses `match`, `|` type unions).
- MIT license file in repo root.
- Package name: `dvrrestorer`. Project root: the repository checkout (same working directory the user is in).
- All imports absolute (`from dvrrestorer.core.scan import ...`), no implicit relative.
- Every `fix_*` function signature is `fn(frame_bgr, alpha_mask, **params) -> frame_bgr`. `alpha_mask` is a `float32` HxW array in `[0, 1]` where 1 = OSD (keep original), 0 = fix freely. Soft mask.
- No ML models in v1 — baseline-only pipeline. ONNX plumbing is designed in but no model is loaded yet (that's a Phase-2 plan).
- All timestamps in seconds (`float`), all frame indices zero-based `int`.
- No committed binary fixtures — synthetic clips are generated on demand by `scripts/synthesize_dvr_artifacts.py`.
- CPU-only dev target; `onnxruntime.get_available_providers()` is checked but no GPU-specific code paths yet.
- Project uses `git` from Task 1 onwards; every task ends with a commit.
- Tests use `pytest`; fixtures in `tests/fixtures/` are built lazily by a session-scoped fixture, never committed.
- All ffmpeg invocations use the `imageio_ffmpeg.get_ffmpeg_exe()` binary, never the system's `ffmpeg`.

## Review Focus

The five input classes/failure modes the spec implies but no task's happy-path test exercises, that are most likely to bite a real user. Each has a pinned test in the owning task.

1. **Variable-fps container (`.mov` with VFR, or GoPro mp4)** — `cv2.VideoCapture.get(CAP_PROP_FPS)` returns the container's declared fps, not the actual frame cadence; timestamp math breaks. Covered by test in **Task 3** (`test_video_io_vfr_timing`).
2. **Non-ASCII / spaces in file paths** — ffmpeg invoked via subprocess breaks on cp1252 systems if argv isn't utf-8. Covered by test in **Task 3** (`test_video_io_unicode_path_roundtrip`).
3. **Clip without an audio stream** — audio-remux step crashes if no audio exists. Covered by test in **Task 3** (`test_video_io_remux_no_audio_source`).
4. **Clip shorter than the zone-detector minimum (<2 s)** — zone segmentation and keyframe-interpolation both assume ≥2 keyframes. Covered by test in **Task 10** (`test_zone_detector_short_clip`).
5. **Keyframes at identical t_sec, or outside the clip range** — interpolation would NaN or extrapolate wrong. Covered by test in **Task 9** (`test_pipeline_keyframe_edge_cases`).

---

### Task 1: Project Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE` (MIT, copyright "Froscht 2026")
- Create: `README.md` (stub: project name, one-paragraph description, "see docs/")
- Create: `.gitignore` (python standard + `.superpowers/` + `tests/fixtures/*.mp4`)
- Create: `dvrrestorer/__init__.py` (just `__version__ = "0.1.0.dev0"`)
- Create: `dvrrestorer/core/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_smoke.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: `dvrrestorer.__version__: str`, package importable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports_and_has_version():
    import dvrrestorer
    assert isinstance(dvrrestorer.__version__, str)
    assert dvrrestorer.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL with ModuleNotFoundError (package not installed).

- [ ] **Step 3: Create `pyproject.toml` with hatchling backend, project.name = "dvrrestorer", requires-python = ">=3.11", dependencies = ["opencv-python>=4.9", "numpy>=1.24", "scipy>=1.11", "onnxruntime>=1.17", "imageio-ffmpeg>=0.4"], dev-dependency pytest. Set `[project.scripts] dvrrestorer-cli = "dvrrestorer.cli:main"`. Add optional-dependencies: `cuda = ["onnxruntime-gpu"]`, `dml = ["onnxruntime-directml"]`, `coreml = ["onnxruntime-coreml"]`.**

- [ ] **Step 4: Create the package skeleton files and the README/LICENSE/.gitignore.**

- [ ] **Step 5: Run `git init`, `pip install -e .`, then re-run the test.**

Run: `cd <repo-root> && git init && pip install -e . && pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat: bootstrap dvrrestorer package"
```

---

### Task 2: Synthetic DVR Artifact Generator

**Files:**
- Create: `scripts/synthesize_dvr_artifacts.py`
- Create: `tests/test_synth.py`
- Create: `tests/conftest.py` (session fixture that generates test clips lazily)

**Interfaces:**
- Produces (module-level in `scripts/synthesize_dvr_artifacts.py`):
  - `make_clean_clip(path: Path, duration_s: float = 10.0, fps: float = 30.0, size: tuple[int, int] = (960, 540)) -> None`
  - `add_chroma_noise(frame: np.ndarray, strength: float) -> np.ndarray`
  - `add_rf_streaks(frame: np.ndarray, strength: float, angle_deg: float = 0.0) -> np.ndarray`
  - `darken(frame: np.ndarray, gamma: float) -> np.ndarray`
  - `make_artifacted_clip(path: Path, **kwargs) -> None` (wraps the three above with per-time-range control)
- Produces (`tests/conftest.py`):
  - pytest fixture `synth_clip_clean(tmp_path_factory) -> Path`
  - pytest fixture `synth_clip_noisy(tmp_path_factory) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synth.py
import cv2
def test_make_clean_clip_writes_readable_mp4(tmp_path):
    from scripts.synthesize_dvr_artifacts import make_clean_clip
    out = tmp_path / "clean.mp4"
    make_clean_clip(out, duration_s=1.0, fps=30.0, size=(320, 180))
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30

def test_add_chroma_noise_increases_chroma_std():
    import numpy as np
    from scripts.synthesize_dvr_artifacts import add_chroma_noise
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    yuv_before = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    noisy = add_chroma_noise(frame, strength=0.5)
    yuv_after = cv2.cvtColor(noisy, cv2.COLOR_BGR2YUV)
    assert yuv_after[:, :, 1].std() > yuv_before[:, :, 1].std() + 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synth.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `make_clean_clip` using `cv2.VideoWriter` with `mp4v` fourcc, drawing a moving gradient + a few static shapes so the clip has structure. Implement the three artifact-adding helpers:**

- `add_chroma_noise`: convert to YUV, add gaussian noise `N(0, strength*25)` to U and V channels only, clip, back to BGR.
- `add_rf_streaks`: generate horizontal sinusoidal pattern across rows, rotate by `angle_deg`, add to Y channel with `strength*30` amplitude.
- `darken`: gamma-adjust via LUT, `gamma = 1.0 + 2.0*strength`.

- [ ] **Step 4: Implement `make_artifacted_clip` that takes `duration_s`, `fps`, `size`, and a list of `(t_start, t_end, noise, streak, dark)` tuples; writes per-frame with the three helpers applied where appropriate.**

- [ ] **Step 5: Add `conftest.py` with session-scoped fixtures that generate the clips to `tmp_path_factory.mktemp("synth")` on first use.**

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_synth.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/ tests/
git commit -m "feat: synthetic DVR artifact generator for test fixtures"
```

---

### Task 3: Video I/O

**Files:**
- Create: `dvrrestorer/core/video_io.py`
- Create: `tests/test_video_io.py`

**Interfaces:**
- Consumes: synthetic fixtures from Task 2.
- Produces:
  - `@dataclass VideoInfo { path: Path, fps: float, frame_count: int, width: int, height: int, has_audio: bool }`
  - `probe(path: Path) -> VideoInfo`
  - `decode_frames(path: Path, start: int = 0, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]` yielding `(frame_index, bgr_frame)`
  - `class FrameWriter`: context-manager, `__init__(path, fps, size, crf=18)`, `write(frame_bgr)`, `close()`
  - `remux_audio(video_only_path: Path, audio_source: Path, out_path: Path) -> None`
  - `stack_compare(before_path: Path, after_path: Path, out_path: Path) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_video_io.py
def test_probe_returns_correct_metadata(synth_clip_clean):
    from dvrrestorer.core.video_io import probe
    info = probe(synth_clip_clean)
    assert info.fps == 30.0
    assert info.frame_count == 300  # synth_clip_clean is 10s @ 30fps
    assert info.width > 0 and info.height > 0

def test_decode_frames_yields_expected_count(synth_clip_clean):
    from dvrrestorer.core.video_io import decode_frames
    frames = list(decode_frames(synth_clip_clean))
    assert len(frames) == 300
    assert frames[0][0] == 0
    assert frames[-1][0] == 299
    assert frames[0][1].ndim == 3 and frames[0][1].shape[2] == 3

def test_frame_writer_roundtrip(tmp_path, synth_clip_clean):
    from dvrrestorer.core.video_io import decode_frames, FrameWriter, probe
    out = tmp_path / "roundtrip.mp4"
    info = probe(synth_clip_clean)
    with FrameWriter(out, fps=info.fps, size=(info.width, info.height)) as w:
        for _, f in decode_frames(synth_clip_clean):
            w.write(f)
    assert out.exists() and out.stat().st_size > 1000
    assert probe(out).frame_count == info.frame_count

def test_video_io_vfr_timing():
    # Review-focus pin #1: variable-fps container
    # Build a tiny clip with mismatched container fps using ffmpeg -framerate,
    # confirm probe() returns the container's declared fps (document the
    # behavior) rather than crashing.
    ...

def test_video_io_unicode_path_roundtrip(tmp_path, synth_clip_clean):
    # Review-focus pin #2: non-ASCII paths
    weird = tmp_path / "ünïcödé with space.mp4"
    from dvrrestorer.core.video_io import decode_frames, FrameWriter, probe
    with FrameWriter(weird, fps=30.0, size=(320, 180)) as w:
        for _, f in decode_frames(synth_clip_clean):
            w.write(f)
            break
    assert weird.exists()
    assert probe(weird).frame_count >= 1

def test_video_io_remux_no_audio_source(tmp_path, synth_clip_clean):
    # Review-focus pin #3: audio remux when source has no audio
    from dvrrestorer.core.video_io import remux_audio
    out = tmp_path / "remuxed.mp4"
    # Should not crash; should produce a valid video-only file.
    remux_audio(synth_clip_clean, synth_clip_clean, out)
    assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_video_io.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `probe` and `decode_frames` using `cv2.VideoCapture`. `probe.has_audio` is determined by shelling out to `ffprobe` (via `imageio_ffmpeg.get_ffmpeg_exe()` sibling path, or parse `ffmpeg -i` stderr).**

- [ ] **Step 4: Implement `FrameWriter`. Prefer piping frames to `ffmpeg -y -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i - -c:v libx264 -crf CRF -preset medium out.mp4` via subprocess; this avoids OpenCV's codec limitations and gives real CRF control. Use `subprocess.Popen` with stdin=PIPE, write raw frame bytes, close stdin on `__exit__`.**

- [ ] **Step 5: Implement `remux_audio` by invoking ffmpeg: `-i video_only -i audio_source -c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0? out`. The `?` makes the audio stream optional — covers Review-Focus #3.**

- [ ] **Step 6: Implement `stack_compare` by invoking ffmpeg with the `hstack` filter.**

- [ ] **Step 7: All subprocess calls that take paths must use the `str(path)` form in argv (not shell-concat); on Windows, set `creationflags=subprocess.CREATE_NO_WINDOW` to suppress console flashes.**

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_video_io.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add dvrrestorer/core/video_io.py tests/test_video_io.py
git commit -m "feat(core): video_io with ffmpeg subprocess, VFR and Unicode-safe"
```

---

### Task 4: Scan (Per-Frame Metrics)

**Files:**
- Create: `dvrrestorer/core/scan.py`
- Create: `tests/test_scan.py`

**Interfaces:**
- Consumes: `decode_frames` from Task 3.
- Produces:
  - `@dataclass FrameMetrics { frame: int, t_sec: float, mean_lum: float, chroma_noise: float, row_streak: float, sharpness: float, black_ratio: float, white_ratio: float, freeze_mad: float | None }`
  - `scan(path: Path, sample_every: int = 2) -> list[FrameMetrics]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan.py
def test_scan_clean_clip_has_low_noise_and_streak(synth_clip_clean):
    from dvrrestorer.core.scan import scan
    rows = scan(synth_clip_clean, sample_every=5)
    assert len(rows) > 0
    assert max(r.chroma_noise for r in rows) < 3.0
    assert max(r.row_streak for r in rows) < 5.0

def test_scan_noisy_clip_detects_peaks(synth_clip_noisy):
    # synth_clip_noisy has chroma noise 0.6 added in a known time range (3-5s)
    from dvrrestorer.core.scan import scan
    rows = scan(synth_clip_noisy, sample_every=5)
    noisy = [r for r in rows if 3.0 <= r.t_sec <= 5.0]
    quiet = [r for r in rows if r.t_sec < 2.0]
    assert max(r.chroma_noise for r in noisy) > 2 * max(r.chroma_noise for r in quiet)
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `pytest tests/test_scan.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scan`. For each sampled frame, downscale to 480×270 for metric computation (scale-invariant for the measures we use). Compute `mean_lum`, `chroma_noise`, `row_streak`, `sharpness`, `black_ratio`, `white_ratio` and `freeze_mad` per frame.**

Specifically:
- `mean_lum` = mean of grayscale.
- `chroma_noise` = mean of `hypot(U - gaussblur(U, σ=3), V - gaussblur(V, σ=3))`.
- `row_streak` = 95th percentile of absolute row-mean differences in the upper half.
- `sharpness` = variance of Laplacian.
- `black_ratio`, `white_ratio` = fraction of pixels with gray <10 or >245.
- `freeze_mad` = mean absolute diff vs. previous sampled gray frame; `None` for first.

- [ ] **Step 4: Run tests to verify they pass.**

Run: `pytest tests/test_scan.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/core/scan.py tests/test_scan.py
git commit -m "feat(core): per-frame scan metrics"
```

---

### Task 5: OSD Detection & Soft-Alpha Mask

**Files:**
- Create: `dvrrestorer/core/osd.py`
- Create: `tests/test_osd.py`
- Modify: `scripts/synthesize_dvr_artifacts.py` — add helper `draw_fake_osd(frame, text="15.5V", pos=(0.75, 0.3))` and expose a fixture `synth_clip_with_osd`.

**Interfaces:**
- Consumes: `decode_frames` (Task 3).
- Produces:
  - `detect_osd_mask(path: Path, sample_every: int = 5, max_samples: int = 60) -> np.ndarray` — returns a `float32` HxW soft-alpha mask in `[0, 1]`.
  - `apply_soft_dilation(mask_bool: np.ndarray, dilate_px: int = 3, blur_sigma: float = 1.5) -> np.ndarray` — binary mask to soft alpha.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_osd.py
def test_detect_osd_mask_finds_painted_osd(synth_clip_with_osd):
    from dvrrestorer.core.osd import detect_osd_mask
    mask = detect_osd_mask(synth_clip_with_osd)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    # The painted OSD is in the top-right quadrant.
    h, w = mask.shape
    tr = mask[: h // 2, w // 2 :]
    rest = mask.copy()
    rest[: h // 2, w // 2 :] = 0
    assert tr.mean() > rest.mean() * 3

def test_apply_soft_dilation_grows_and_feathers():
    from dvrrestorer.core.osd import apply_soft_dilation
    bm = np.zeros((100, 100), dtype=bool)
    bm[40:60, 40:60] = True
    soft = apply_soft_dilation(bm, dilate_px=3, blur_sigma=1.5)
    assert soft.shape == (100, 100)
    assert soft[50, 50] > 0.95
    assert soft[0, 0] < 0.05
    # Edge pixel should be between
    edge = soft[62, 50]
    assert 0.0 < edge < 1.0
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `pytest tests/test_osd.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `apply_soft_dilation` using `cv2.dilate` + `cv2.GaussianBlur` on a float32 cast.**

- [ ] **Step 4: Implement `detect_osd_mask`:**
1. Sample `max_samples` frames evenly from the clip.
2. For each sample, run `cv2.Canny` on grayscale, dilate 1 px → per-frame edge map.
3. Average all edge maps; threshold at 0.6 of max → candidate OSD region (pixels where edges are present in >60% of samples).
4. Pass the boolean mask through `apply_soft_dilation(..., dilate_px=3, blur_sigma=1.5)`.

- [ ] **Step 5: Add `draw_fake_osd` to the synthesis script and a `synth_clip_with_osd` fixture in `conftest.py`.**

- [ ] **Step 6: Run tests to verify they pass.**

Run: `pytest tests/test_osd.py tests/test_synth.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add dvrrestorer/core/osd.py tests/test_osd.py scripts/synthesize_dvr_artifacts.py tests/conftest.py
git commit -m "feat(core): heuristic OSD detection with soft-alpha mask"
```

---

### Task 6: Dark-Fix (Gamma / CLAHE Baseline)

**Files:**
- Create: `dvrrestorer/core/fix_dark.py`
- Create: `tests/test_fix_dark.py`

**Interfaces:**
- Produces:
  - `fix_dark(frame_bgr: np.ndarray, alpha_mask: np.ndarray | None, strength: float, shadow_boost: float = 0.5) -> np.ndarray`
    - `strength ∈ [0, 1]`; `0` = no-op, `1` = aggressive. `shadow_boost ∈ [0, 1]` biases toward lifting shadows vs. midtones.
    - When `alpha_mask` is not None, output is blended with the input via `(1 - alpha_mask)` so OSD pixels pass through unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fix_dark.py
def test_fix_dark_zero_strength_is_identity():
    import numpy as np
    from dvrrestorer.core.fix_dark import fix_dark
    f = (np.random.rand(50, 80, 3) * 100).astype(np.uint8)
    out = fix_dark(f, None, strength=0.0)
    assert (out == f).all()

def test_fix_dark_increases_mean_luminance_for_dark_input():
    import numpy as np, cv2
    from dvrrestorer.core.fix_dark import fix_dark
    f = np.full((50, 80, 3), 20, dtype=np.uint8)
    out = fix_dark(f, None, strength=0.8)
    assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean() > 20 + 20

def test_fix_dark_respects_osd_mask():
    import numpy as np
    from dvrrestorer.core.fix_dark import fix_dark
    f = np.full((50, 80, 3), 20, dtype=np.uint8)
    mask = np.zeros((50, 80), dtype=np.float32)
    mask[:10, :10] = 1.0  # top-left is OSD, must stay 20
    out = fix_dark(f, mask, strength=0.8)
    assert out[5, 5].mean() == 20
    assert out[40, 70].mean() > 25
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `fix_dark` using a `strength`-scaled gamma curve (`gamma = 1 - strength * 0.7`, so strength=0 → gamma=1, strength=1 → gamma=0.3) applied via LUT, then CLAHE on L channel (LAB) with `clipLimit = 1 + strength * 2`, blended with `shadow_boost` toward the pure-gamma result.**

- [ ] **Step 4: Blend result with original via `alpha_mask` if provided: `out = result * (1 - alpha_mask[..., None]) + frame * alpha_mask[..., None]`.**

- [ ] **Step 5: Run tests to verify they pass.**

- [ ] **Step 6: Commit**

```bash
git add dvrrestorer/core/fix_dark.py tests/test_fix_dark.py
git commit -m "feat(core): fix_dark gamma+CLAHE baseline with OSD alpha-mask"
```

---

### Task 7: Line/Streak Fix (FFT-Wedge)

**Files:**
- Create: `dvrrestorer/core/fix_lines.py`
- Create: `tests/test_fix_lines.py`

**Interfaces:**
- Produces:
  - `fix_lines(frame_bgr: np.ndarray, alpha_mask: np.ndarray | None, strength: float, direction_deg: float = 0.0) -> np.ndarray`
    - Removes periodic streak patterns oriented at `direction_deg` (0 = horizontal).
    - `strength ∈ [0, 1]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fix_lines.py
def test_fix_lines_zero_strength_identity():
    import numpy as np
    from dvrrestorer.core.fix_lines import fix_lines
    f = (np.random.rand(60, 80, 3) * 255).astype(np.uint8)
    out = fix_lines(f, None, strength=0.0)
    assert (out == f).all()

def test_fix_lines_reduces_horizontal_stripe_energy():
    import numpy as np
    from dvrrestorer.core.fix_lines import fix_lines
    from scripts.synthesize_dvr_artifacts import add_rf_streaks
    base = np.full((120, 160, 3), 128, dtype=np.uint8)
    noisy = add_rf_streaks(base, strength=0.8, angle_deg=0.0)
    cleaned = fix_lines(noisy, None, strength=0.7, direction_deg=0.0)
    # Energy in horizontal-stripe FFT band must drop.
    def streak_energy(img):
        g = img[:, :, 1].astype(np.float32)
        F = np.fft.fft2(g)
        return float(np.abs(F[1:5, :]).sum())
    assert streak_energy(cleaned) < streak_energy(noisy) * 0.5
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `fix_lines`:**
1. Convert to YUV; work only on Y.
2. Rotate frame by `-direction_deg` so target streaks become horizontal.
3. 1-D FFT per column → detect horizontal-frequency peaks.
4. Build a wedge-shaped notch mask around detected peak frequencies, scaled by `strength` (strength=1 = narrower notch, more aggressive removal; strength=0.1 = very mild).
5. Apply inverse FFT, rotate back, merge Y with original U/V, convert back to BGR.
6. Apply alpha-mask blend as in Task 6.

- [ ] **Step 4: Run tests to verify they pass.**

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/core/fix_lines.py tests/test_fix_lines.py
git commit -m "feat(core): fix_lines FFT-wedge destripe with OSD alpha-mask"
```

---

### Task 8: Chroma-Noise Fix (NLM + Flow Temporal Median)

**Files:**
- Create: `dvrrestorer/core/fix_chroma.py`
- Create: `tests/test_fix_chroma.py`

**Interfaces:**
- Produces:
  - `fix_chroma(frame_bgr: np.ndarray, prev_frames: list[np.ndarray], next_frames: list[np.ndarray], alpha_mask: np.ndarray | None, strength: float, temporal_w: float = 0.5) -> np.ndarray`
    - `prev_frames`, `next_frames` can be empty — then purely spatial.
    - `strength ∈ [0, 1]`, `temporal_w ∈ [0, 1]` (0 = pure spatial NLM, 1 = pure temporal median).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fix_chroma.py
def test_fix_chroma_zero_strength_identity():
    import numpy as np
    from dvrrestorer.core.fix_chroma import fix_chroma
    f = (np.random.rand(50, 80, 3) * 255).astype(np.uint8)
    out = fix_chroma(f, [], [], None, strength=0.0)
    assert (out == f).all()

def test_fix_chroma_reduces_chroma_std_on_noisy_input():
    import numpy as np, cv2
    from dvrrestorer.core.fix_chroma import fix_chroma
    from scripts.synthesize_dvr_artifacts import add_chroma_noise
    base = np.full((80, 100, 3), 128, dtype=np.uint8)
    noisy = add_chroma_noise(base, strength=0.7)
    cleaned = fix_chroma(noisy, [], [], None, strength=0.7)
    def chroma_std(img):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 1].std() + yuv[:, :, 2].std()
    assert chroma_std(cleaned) < chroma_std(noisy) * 0.5
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `fix_chroma`:**
1. Spatial path: convert to YUV, run `cv2.fastNlMeansDenoisingColored` on BGR with `h=strength*10, hColor=strength*10` — call it only if `strength > 0`.
2. Temporal path (only if `prev_frames + next_frames` nonempty and `temporal_w > 0`):
   - Compute optical flow via `cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)` from current to each neighbor, warp neighbor to current frame coords with `cv2.remap`.
   - Stack warped neighbors + current, take per-pixel median over U and V channels only.
3. Blend spatial and temporal results by `temporal_w`.
4. Apply alpha-mask blend.

- [ ] **Step 4: Run tests to verify they pass.**

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/core/fix_chroma.py tests/test_fix_chroma.py
git commit -m "feat(core): fix_chroma NLM + flow-median temporal denoise"
```

---

### Task 9: Pipeline (Chunked, OSD-aware)

**Files:**
- Create: `dvrrestorer/core/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 3–8.
- Produces:
  - `@dataclass Keyframe { t_sec: float, dark: float, lines: float, chroma: float }`
  - `@dataclass PipelineConfig { keyframes: list[Keyframe], chunk_size: int = 32, chroma_lookahead: int = 2, direction_deg: float = 0.0 }`
  - `interpolate(keyframes: list[Keyframe], t_sec: float) -> tuple[float, float, float]` — linear per track, clamp outside range to the endpoint value, duplicate `t_sec`s take the later one.
  - `run(input_path: Path, output_path: Path, config: PipelineConfig, osd_mask: np.ndarray | None = None, progress_cb: Callable[[int, int], None] | None = None) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
def test_interpolate_at_keyframe_exact():
    from dvrrestorer.core.pipeline import Keyframe, interpolate
    kfs = [Keyframe(0.0, 0.1, 0.2, 0.3), Keyframe(2.0, 0.5, 0.6, 0.7)]
    assert interpolate(kfs, 0.0) == (0.1, 0.2, 0.3)
    assert interpolate(kfs, 2.0) == (0.5, 0.6, 0.7)

def test_interpolate_linear_midpoint():
    from dvrrestorer.core.pipeline import Keyframe, interpolate
    kfs = [Keyframe(0.0, 0.0, 0.0, 0.0), Keyframe(2.0, 1.0, 1.0, 1.0)]
    assert interpolate(kfs, 1.0) == (0.5, 0.5, 0.5)

def test_pipeline_keyframe_edge_cases():
    # Review-focus pin #5
    from dvrrestorer.core.pipeline import Keyframe, interpolate
    # outside range -> clamp
    kfs = [Keyframe(1.0, 0.3, 0.3, 0.3), Keyframe(2.0, 0.7, 0.7, 0.7)]
    assert interpolate(kfs, 0.0) == (0.3, 0.3, 0.3)
    assert interpolate(kfs, 5.0) == (0.7, 0.7, 0.7)
    # duplicate t_sec -> later wins
    kfs = [Keyframe(1.0, 0.1, 0.1, 0.1), Keyframe(1.0, 0.9, 0.9, 0.9)]
    assert interpolate(kfs, 1.0) == (0.9, 0.9, 0.9)

def test_pipeline_runs_end_to_end(synth_clip_noisy, tmp_path):
    from dvrrestorer.core.pipeline import run, PipelineConfig, Keyframe
    out = tmp_path / "cleaned.mp4"
    cfg = PipelineConfig(keyframes=[Keyframe(0.0, 0.5, 0.5, 0.5)])
    run(synth_clip_noisy, out, cfg)
    assert out.exists() and out.stat().st_size > 1000
    # Frame count preserved.
    from dvrrestorer.core.video_io import probe
    assert probe(out).frame_count == probe(synth_clip_noisy).frame_count
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `interpolate`. Sort keyframes by `t_sec` stably; for a duplicate `t_sec` the latest-added wins (iterate from end for lookup).**

- [ ] **Step 4: Implement `run`:**
1. `probe` input to get fps/frame_count.
2. If `osd_mask is None`, compute it via `osd.detect_osd_mask(input_path)`.
3. Open `FrameWriter` on a temporary video-only path.
4. For each chunk of `chunk_size` frames:
   - Decode the chunk plus `chroma_lookahead` extra at each end (for temporal context).
   - For each frame in the chunk, interpolate params at `t_sec = frame_idx / fps`, apply `fix_dark → fix_lines → fix_chroma`, write to the FrameWriter.
5. Close the writer.
6. `remux_audio(video_only_path, input_path, output_path)` to copy original audio in (handles no-audio case).
7. Call `progress_cb(current, total)` once per frame if provided.

- [ ] **Step 5: Run tests to verify they pass.**

- [ ] **Step 6: Commit**

```bash
git add dvrrestorer/core/pipeline.py tests/test_pipeline.py
git commit -m "feat(core): chunked OSD-aware pipeline with keyframe interpolation"
```

---

### Task 10: Zone Detector

**Files:**
- Create: `dvrrestorer/core/zone_detector.py`
- Create: `tests/test_zone_detector.py`

**Interfaces:**
- Consumes: `scan` from Task 4, `Keyframe` from Task 9.
- Produces:
  - `@dataclass Zone { t_start: float, t_end: float, dominant: str, metrics_summary: dict[str, float] }`
  - `detect_zones(metrics: list[FrameMetrics], min_zone_len_s: float = 1.0, merge_gap_s: float = 0.5, max_keyframes: int = 20) -> list[Zone]`
  - `auto_keyframes(zones: list[Zone], clip_duration_s: float) -> list[Keyframe]` — generates one initial keyframe per zone with values derived from its `metrics_summary`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_zone_detector.py
def test_detect_zones_on_noisy_clip(synth_clip_noisy):
    from dvrrestorer.core.scan import scan
    from dvrrestorer.core.zone_detector import detect_zones
    rows = scan(synth_clip_noisy, sample_every=3)
    zones = detect_zones(rows)
    assert len(zones) >= 1
    noisy_zone = next((z for z in zones if 3.0 <= z.t_start <= 5.5), None)
    assert noisy_zone is not None
    assert noisy_zone.dominant in {"chroma_noise", "mixed"}

def test_auto_keyframes_count_bounded(synth_clip_noisy):
    from dvrrestorer.core.scan import scan
    from dvrrestorer.core.zone_detector import detect_zones, auto_keyframes
    rows = scan(synth_clip_noisy, sample_every=3)
    zones = detect_zones(rows)
    kfs = auto_keyframes(zones, clip_duration_s=10.0)
    assert len(kfs) <= 20
    # At least one keyframe must sit in the noisy region.
    assert any(3.0 <= k.t_sec <= 5.5 and k.chroma > 0.3 for k in kfs)

def test_zone_detector_short_clip():
    # Review-focus pin #4
    from dvrrestorer.core.scan import FrameMetrics
    from dvrrestorer.core.zone_detector import detect_zones, auto_keyframes
    rows = [FrameMetrics(frame=i, t_sec=i*0.1, mean_lum=50.0, chroma_noise=1.0,
                         row_streak=1.0, sharpness=100.0, black_ratio=0.0,
                         white_ratio=0.0, freeze_mad=None) for i in range(10)]  # 1-second clip
    zones = detect_zones(rows, min_zone_len_s=1.0)
    kfs = auto_keyframes(zones, clip_duration_s=1.0)
    # Must produce at least a start- and end-keyframe so interpolation works.
    assert len(kfs) >= 2
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `detect_zones`:**
1. For each metric (`chroma_noise`, `row_streak`, low `mean_lum`), threshold at the 70th percentile over the clip → per-frame boolean "problem flag" per metric.
2. Scan for runs of flag-set frames; collapse runs shorter than `min_zone_len_s` into neighbors or drop them; merge adjacent runs with a gap < `merge_gap_s`.
3. For each zone, record which metric dominated (highest percentile exceedance), or `"mixed"`.

- [ ] **Step 4: Implement `auto_keyframes`:**
1. Always emit a keyframe at `t_sec = 0` and `t_sec = clip_duration_s` with "quiet" default values (0.2 for each track).
2. For each zone, emit a keyframe at its midpoint with values derived from `metrics_summary` (e.g., chroma = clip(chroma_noise_mean / 10, 0.2, 1.0)).
3. Trim to `max_keyframes` by keeping the strongest (highest metric values).

- [ ] **Step 5: Run tests to verify they pass.**

- [ ] **Step 6: Commit**

```bash
git add dvrrestorer/core/zone_detector.py tests/test_zone_detector.py
git commit -m "feat(core): zone detection and auto-keyframe placement"
```

---

### Task 11: Profiles (JSON Preset I/O)

**Files:**
- Create: `dvrrestorer/profiles.py`
- Create: `tests/test_profiles.py`

**Interfaces:**
- Produces:
  - `save_profile(path: Path, name: str, keyframes: list[Keyframe], direction_deg: float = 0.0) -> None`
  - `load_profile(path: Path) -> tuple[str, list[Keyframe], float]`
  - `list_profiles(dir_path: Path) -> list[str]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_profiles.py
def test_save_load_roundtrip(tmp_path):
    from dvrrestorer.profiles import save_profile, load_profile
    from dvrrestorer.core.pipeline import Keyframe
    kfs = [Keyframe(0.0, 0.1, 0.2, 0.3), Keyframe(2.5, 0.5, 0.6, 0.7)]
    p = tmp_path / "my.json"
    save_profile(p, "test", kfs, direction_deg=5.0)
    name, loaded, dir_deg = load_profile(p)
    assert name == "test"
    assert loaded == kfs
    assert dir_deg == 5.0

def test_list_profiles_sorted(tmp_path):
    from dvrrestorer.profiles import save_profile, list_profiles
    from dvrrestorer.core.pipeline import Keyframe
    for n in ("beta", "alpha", "gamma"):
        save_profile(tmp_path / f"{n}.json", n, [Keyframe(0, 0, 0, 0)])
    assert list_profiles(tmp_path) == ["alpha", "beta", "gamma"]
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `save_profile`, `load_profile`, `list_profiles` using `json` module. Schema: `{"name": str, "direction_deg": float, "keyframes": [{"t_sec": float, "dark": float, "lines": float, "chroma": float}, ...]}`. Include a schema version `"version": 1`.**

- [ ] **Step 4: Run tests to verify they pass.**

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/profiles.py tests/test_profiles.py
git commit -m "feat: profile JSON persistence"
```

---

### Task 12: CLI (Headless Entry-Point)

**Files:**
- Create: `dvrrestorer/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 3–11.
- Produces:
  - `main(argv: list[str] | None = None) -> int` — exit code.

CLI shape:
```
dvrrestorer-cli INPUT [--out PATH] [--preset PATH] [--auto]
                       [--compare] [--chunk-size N] [--direction-deg DEG]
```
- `--auto` runs `scan` + `zone_detector` + `auto_keyframes` to build a default timeline.
- `--preset` loads a saved profile. Mutually exclusive with `--auto`.
- `--compare` additionally writes `<out>.compare.mp4` via `stack_compare`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
def test_cli_auto_produces_output(synth_clip_noisy, tmp_path):
    out = tmp_path / "cleaned.mp4"
    from dvrrestorer.cli import main
    rc = main([str(synth_clip_noisy), "--out", str(out), "--auto"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 1000

def test_cli_preset_mutually_exclusive_with_auto(synth_clip_noisy, tmp_path):
    from dvrrestorer.cli import main
    import pytest
    with pytest.raises(SystemExit):
        main([str(synth_clip_noisy), "--preset", "x.json", "--auto"])
```

- [ ] **Step 2: Run tests to verify they fail.**

- [ ] **Step 3: Implement `main` with `argparse`. Default `--out` is `<input_stem>_cleaned.mp4` next to the input. Preset path or `--auto` is required (ArgParse mutually exclusive group).**

- [ ] **Step 4: Wire `main` to call `pipeline.run` with the resolved keyframes. Install a progress-print callback that writes `done/total` every 30 frames to stderr.**

- [ ] **Step 5: Run tests to verify they pass.**

- [ ] **Step 6: Smoke-check manually: run the installed script.**

Run (shell):
```
dvrrestorer-cli "INPUT.mp4" --auto --compare --out "%TEMP%\fpv_cleaned.mp4"
```
Expected: completes without error, writes both outputs, prints progress.

- [ ] **Step 7: Commit**

```bash
git add dvrrestorer/cli.py tests/test_cli.py
git commit -m "feat(cli): headless entry-point with --auto and --preset"
```

---

## Final Verification

After Task 12:

- [ ] Run full test suite: `pytest -v`
- [ ] Verify every `test_*` passes.
- [ ] Verify the five Review-Focus pin tests exist and pass:
  - `test_video_io_vfr_timing` (Task 3)
  - `test_video_io_unicode_path_roundtrip` (Task 3)
  - `test_video_io_remux_no_audio_source` (Task 3)
  - `test_zone_detector_short_clip` (Task 10)
  - `test_pipeline_keyframe_edge_cases` (Task 9)
- [ ] Run the manual smoke on the real YouTube-sourced FPV clip and confirm:
  - A `_cleaned.mp4` exists.
  - It plays end-to-end.
  - The dark baumschatten regions look lifted.
  - OSD at the top-right ("15.5V", "02:37") looks identical to the input.
- [ ] Final commit `chore: v0.1.0 MVP core + CLI`.

Next: separate plan for the GUI (Timeline widget, Preview pane, Params pane, app wiring) sitting on top of this MVP.

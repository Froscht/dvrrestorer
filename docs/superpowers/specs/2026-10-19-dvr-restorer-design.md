# FPV DVR Restorer — Design

**Status:** draft • **Datum:** 2026-10-19 • **Autor:** Froscht

## Intent und Why

FPV-Piloten nehmen ihre Flüge am VRX/Goggle als DVR-Datei auf. Das Analog-Signal (5.8 GHz CVBS) bringt im DVR drei dominante Artefakte:

1. **Chroma-Rauschen** flächig über das Bild wenn das RF-Signal schwach wird (Entfernung, Hindernisse)
2. **RF-Streifen** (horizontale/diagonale Beugungsmuster, besonders in homogenen Bildteilen wie Himmel)
3. **Dunkle Bildbereiche** die im DVR praktisch nicht mehr lesbar sind (Baumschatten, Sonnenstand)

Bestehende Werkzeuge in diesem Nischenbereich sind kommerziell, Windows-only und arbeiten mit starren Presets. Ein freier, moderner, cross-platform Ersatz existiert nicht.

Dieses Projekt baut eine **Open-Source-Alternative** mit vier klaren Zielen: Timeline-Keyframe-UI statt Preset-Wahl, ML-Denoiser (NAFNet/SCUNet) zusätzlich zur klassischen Pipeline, Zero-DCE/RetinexNet für Low-Light statt reiner Gamma-Curve, und cross-platform.

## Goals

- **G1:** Clip rein → repariertes Clip raus für die drei Artefaktklassen oben
- **G2:** **Timeline-GUI** mit Vorschau-Preview (<100 ms pro Frame), automatisch erkannten Problem-Zonen als vorbesetzte Keyframes, pro-Keyframe-Regler für Dunkel-Fix, Streifen, Chroma
- **G3:** **Hybrid-Pipeline:** klassische Methoden (FFT-Destripe, Flow-Inpaint, Gamma) als Baseline; **optionale ML-Blöcke** (NAFNet/SCUNet, Zero-DCE/RetinexNet) für harte Fälle, als ONNX-Modelle austauschbar
- **G4:** **OSD-Protection:** eingebranntes OSD (Spannung, Timer, Battery-Status) wird erkannt und bleibt beim Fixen unangetastet
- **G5:** **Cross-platform** (Windows/Linux/macOS), Python 3.11+, **MIT-Lizenz**
- **G6:** GPU-Beschleunigung via ONNX Runtime mit DirectML (Windows), CUDA (NVIDIA), CoreML (Mac) — automatische Backend-Wahl
- **G7:** Export als `_cleaned.mp4` (H.264 via ffmpeg) + optional `_compare.mp4` (Side-by-Side Vorher/Nachher) mit Original-Audio

## Non-Goals (für v1)

- **NG1:** HDZero / DJI / Walksnail digitale FPV-Reparatur. v1 fokussiert auf Analog.
- **NG2:** Super-Resolution / 2x Upscaling. Separates Topic.
- **NG3:** Künstliches Motion Blur
- **NG4:** OSD-Editor — out of scope
- **NG5:** Highlight-Detection / Clip-Trimming / Social-Sharing
- **NG6:** Mobile-App, Web-App, Cloud-Verarbeitung
- **NG7:** Eigenes Training von ML-Modellen. Wir nutzen pretrained ONNX-Modelle von veröffentlichten Papers/Repos; wenn die zu generisch sind, Nachtraining ist späteres Topic.

## Anti-Goals (woran dieses Projekt scheitert, selbst wenn es "funktioniert")

- **AG1:** Wird zum weiteren Preset-Lotterie-Tool. Wenn der Nutzer am Ende doch wieder nur "Profile wählen → warten → schauen" macht, haben wir den Hauptvorteil (Timeline-Tuning) verfehlt.
- **AG2:** ML-Modelle verbrennen den Clip. Wenn NAFNet/SCUNet ohne Deckel laufen und detailreiche Flächen wie Blätter zu Plastik glätten, ist das schlimmer als unverarbeitet.
- **AG3:** CPU-Only-Fallback ist unerträglich langsam. Wenn ohne GPU >10x Realtime vergeht, nutzt niemand das Tool.
- **AG4:** Monolithisches 400-MB-Bundle. Sauberes Package (<80 MB ohne Modelle, Modelle on-demand).

## Constraints

- **C1:** **Keine echten FPV-DVR-Originale** in diesem Repo (Lizenz/Datenschutz der aufnehmenden Piloten). Entwicklung muss mit YouTube-FPV-Clips (Chroma-Rauschen durch Reencode weggefiltert) und synthetischem Test-Material (Noise/Streifen künstlich auf saubere Clips geaddiert) laufen. User testet mit eigenen Originalen.
- **C2:** Nutzer-Hardware ist Windows 11 (bekannt). Linux/Mac als Secondary-Target.
- **C3:** **Python-Stack verpflichtend** (CustomTkinter + OpenCV + onnxruntime + ffmpeg via imageio_ffmpeg) — bewährte Toolkette, cross-platform, breite Community.
- **C4:** Keine kommerziellen Dependencies. Alles MIT/BSD/Apache-kompatibel.
- **C5:** Entwicklungs-GPU unbekannt (muss geklärt werden) → siehe offene Punkte.

## Architektur

```
dvrrestorer/
├── core/                        # pure functions, keine GUI
│   ├── scan.py                  # Frame-Analyse (lum, chroma_noise, row_streak, sharpness, badness)
│   ├── osd.py                   # OSD-Erkennung + Masken (template-matching + edge-density)
│   ├── fix_dark.py              # Dunkel-Fix: Gamma/CLAHE baseline + Zero-DCE ONNX
│   ├── fix_lines.py             # Streifen-Fix: FFT-Wedge baseline + Variational optional
│   ├── fix_chroma.py            # Chroma-Fix: Spatial median baseline + NAFNet ONNX
│   ├── pipeline.py              # Chunked frame-processing, OSD-mask-aware
│   └── video_io.py              # ffmpeg-wrapper (decode, encode, audio-remux, compare-stack)
├── gui/
│   ├── app.py                   # CustomTkinter main window
│   ├── timeline.py              # Zeitleiste mit auto-erkannten Zonen + Keyframes
│   ├── preview.py               # Preview-Pane mit Vorher/Nachher-Toggle
│   ├── params.py                # Regler für aktiven Keyframe
│   └── zone_detector.py         # ruft core/scan auf, segmentiert zu Timeline-Zonen
├── models/                      # ONNX-Modelle, on-demand downloadbar
│   └── README.md                # Herkunft, Lizenzen, Download-URLs
├── profiles/                    # Preset-Dateien (JSON), speicherbare Timeline-Snapshots
├── cli.py                       # headless Batch-Modus (gleicher core, keine GUI)
├── tests/                       # pytest, auf synthetischem Material
├── pyproject.toml
├── LICENSE                      # MIT
└── README.md
```

**Zwei Entry-Points:**
- `dvrrestorer` → startet die GUI
- `dvrrestorer-cli <input> --preset <name> --out <path>` → headless, nutzt gespeicherte Timeline-Presets

### Pipeline-Datenfluss (pro Clip)

```
input.mp4
   │
   ├─> ffmpeg decode → RGB-Frame-Stream
   │
   ├─> scan (vollständiger Durchlauf, 1x) → per-Frame-Metriken + Zonen-Segmentierung
   │
   ├─> osd.detect_from_video → OSD-Maske (meist statisch, 1x berechnet)
   │
   ├─> Timeline-Keyframes (vom User getuned oder Auto-Default)
   │
   ├─> chunked processing (per Chunk 32 Frames):
   │     für jeden Frame:
   │       1. Keyframe-Werte interpolieren
   │       2. fix_dark   (OSD-maskiert)
   │       3. fix_lines  (OSD-maskiert)
   │       4. fix_chroma (OSD-maskiert, temporal-aware: nutzt ±K Nachbarframes)
   │       5. OSD-Pixel aus Originalframe zurückkopieren
   │
   ├─> ffmpeg encode (libx264 CRF 18, oder h264_nvenc wenn verfügbar)
   │
   └─> ffmpeg audio-remux → _cleaned.mp4
```

### ML-Modelle (ONNX, pretrained, austauschbar)

| Zweck | Baseline | ML-Erweiterung | Modell-Quelle |
|-------|----------|----------------|---------------|
| Chroma-Denoise | OpenCV fastNlMeansDenoisingColored + flow-based median über ±2 Frames | **NAFNet** oder **SCUNet** auf chroma-Kanälen | megvii-research/NAFNet (MIT), cszn/SCUNet (MIT) → ONNX export |
| Low-Light | Gamma + CLAHE per-channel | **Zero-DCE** (Zero-Reference Deep Curve Estimation) | Li-Chongyi/Zero-DCE (MIT) |
| Line-Removal | FFT-Wedge-Filter (Standardverfahren gegen periodisches Streifenrauschen) | Variational-Decomposition (klassisch, keine zusätzliche ML) | eigene Implementation |

Modelle liegen **nicht im Repo**, werden beim ersten Lauf heruntergeladen (einfacher HTTP-Download aus `models/README.md`) und per SHA256 verifiziert. Nutzer kann im Settings-Dialog alternative Modelle via Pfad einhängen.

### Timeline + Zone-Detection

- Beim Clip-Laden läuft `scan` einmal durch (sampled jeder N-te Frame, läuft bei 1080p/6min in <10s).
- Metriken: `mean_lum`, `chroma_noise`, `row_streak`, `sharpness`, `black_ratio`, `white_ratio`, `freeze_mad`.
- `zone_detector` segmentiert aufeinanderfolgende Frames mit ähnlichen Metriken zu **Zonen** (ein zusammenhängender Abschnitt mit ähnlichem Problem-Profil, >1 s).
- Jede Zone bekommt einen **initialen Keyframe** mit automatisch gewählten Fix-Stärken basierend auf den Metriken (z.B. hohes chroma_noise → chroma-Fix-Stärke 0.7).
- Nutzer kann Keyframes verschieben, hinzufügen, löschen, Fix-Werte einstellen. Zwischen Keyframes wird linear interpoliert.
- Timeline hat drei horizontale Tracks (Dunkel / Lines / Chroma), jeder Keyframe als Diamant. Marker für erkannte Problem-Zonen als farbige Balken im Hintergrund (orange=dark, lila=streak, türkis=noise).

### Preview-Strategie

- Live-Preview rechnet NUR den aktuellen Frame.
- Alle Fixes müssen **inkrementell** sein: Scrubbt der Nutzer, läuft Pipeline nur für diesen einen Frame.
- Ziel: <100 ms pro Preview-Frame bei 1080p auf Mittelklasse-GPU, <500 ms auf CPU.
- Full-Export rendert alle Frames chunked mit Multithreading + GPU-Batch-Inferenz (Chunk-Size 32).

## Was festgelegt ist (keine weitere Diskussion)

- Python 3.11+ (nicht älter wegen `match` + schnellere starttime)
- CustomTkinter als GUI-Framework (bewährt für Dark-Mode, geringe Install-Last)
- OpenCV 4.x (Bild/Video), numpy, onnxruntime (CPU + GPU)
- imageio-ffmpeg für ffmpeg-Binary (keine Systeminstallation nötig)
- MIT-Lizenz
- GitHub-Hosting (öffentlich)
- Keine Lizenz-/Trial-Mechanik
- Timeline-GUI (vom User bei Mockup-Vergleich A/B gewählt: B)
- v1 nur Analog, HDZero kommt später
- v1 keine Dropouts (zerfetzte komplette Frames) — kommen später wenn echte Beispiele existieren

### Builder-Entscheidungen (nach Review-Pass getroffen)

- **ML-Modelle sind in v1 OPTIONAL.** MVP läuft komplett mit der klassischen Baseline-Pipeline (FFT-Destripe, Flow-Inpaint, Gamma/CLAHE). Grund: pretrained ONNX-Weights für NAFNet/SCUNet/Zero-DCE existieren nicht offiziell — die Repos publizieren PyTorch-Checkpoints. ML-Blöcke werden in Phase 2 nachgezogen, zuerst Zero-DCE (kleinstes Modell, einfachster Export). Jedes Modul (`fix_dark`, `fix_chroma`) wird so gebaut, dass der ML-Teil ein austauschbares Backend ist (`Backend.classical` / `Backend.ml_zero_dce`).
- **onnxruntime als pip-extras.** Default `pip install dvrrestorer` zieht CPU-only. GPU opt-in via `[cuda]` (NVIDIA, Linux/Windows), `[dml]` (DirectML, Windows AMD/Intel) oder `[coreml]` (macOS). PyInstaller-Bundle wird pro Zielplattform einmal mit passendem Backend gebaut.
- **OSD-Detection v1 heuristisch, keine Font-Templates.** `osd.py` nutzt edge-density + bbox-stability über ein Sliding-Window von ~N Frames: Bereiche mit hoher Edge-Density, die über viele Sekunden an derselben Position bleiben, werden als OSD markiert. Deckt die meisten Analog-OSDs ab (sitzen typisch am Rand, sind statisch). Font-Templates und User-Screenshot-Override sind v2.
- **OSD-Mask ist eine Soft-Alpha-Maske**, nicht hart binär. Pipeline: binary mask → dilate (3 px) → gaussian blur (σ=1.5). Verhindert sichtbare Kanten beim Kopieren der OSD-Pixel. Alle `fix_*`-Module nehmen `(frame, alpha_mask)` entgegen und blenden ihr Ergebnis mit `(1 - alpha_mask)` ein.
- **Preview nutzt Baseline-only + Quarter-Res.** Live-Preview rechnet NIE ML-Blöcke (würde das <100-ms-Budget sprengen), nur die klassischen Fixe. Zusätzlich auf Quarter-Res (960×540 statt 1920×1080) + Upscaling zur Anzeige. ML-Blöcke laufen ausschließlich im Full-Export. Preview zeigt deutlich "Preview — Export nutzt zusätzlich ML-Enhancement".
- **Zone-Detector Startwerte:** max 20 Keyframes pro 6-min Clip, min Zone-Länge 1 s, Zonen-Merge wenn <0.5 s Lücke. Werte werden iteriert sobald echte Testclips da sind (siehe OP2).

## Was dem Builder überlassen bleibt

- Konkrete Qt- vs. CTk- vs. andere GUI-Entscheidung sofern CTk sich als zu limitiert für Timeline erweist (Fallback: PySide6)
- Chunk-Size, Thread-Pool-Dimensionierung
- Konkrete Keyframe-Interpolationsart (linear reicht, cubic optional)
- Format der Preset-JSON-Dateien (sinnvolles Schema, erweiterbar)
- Fenster-Layout-Details (Splitter-Verhältnisse, Dark-Theme-Farben)
- Build-/Packaging-Mechanismus (PyInstaller/Briefcase/Nuitka — PyInstaller ist bekannt und tut was wir brauchen)
- Umgang mit sehr großen Clips (>30 min): Streaming-Decode statt alles im RAM, konkrete Grenzen
- Error-Handling-Details (unbekannter Codec, defekter Header, abgebrochener Export)
- Abhängigkeits-Pins im pyproject.toml

## Vorher offene Punkte (vom Nutzer beantwortet)

- **OP1 → GPU-Zielumgebung ist unbekannt** (Endnutzer-abhängig). Konsequenz: Dev-Loop muss komplett CPU-only funktionieren. Die Pipeline ist Backend-agnostisch: `onnxruntime.get_available_providers()` wählt zur Laufzeit (`CUDAExecutionProvider` → `DmlExecutionProvider` → `CoreMLExecutionProvider` → `CPUExecutionProvider`). GPU-Testing passiert nur gegen Zielplattformen in CI oder beim Nutzer.
- **OP2 → Echte DVR-Dateien kommen später.** Konsequenz: MVP wird gegen **synthetisches Test-Material** entwickelt. Dafür wird eine kleine Helper-CLI `scripts/synthesize_dvr_artifacts.py` gebaut, die auf einem sauberen HD-Clip die drei Artefaktklassen (Chroma-Rauschen, RF-Streifen, Low-Light) kontrolliert aufmoduliert. Pipeline-Parameter werden anfänglich daran getuned; sobald echte DVR-Rohdaten verfügbar sind, zweite Tuning-Runde.
- **OP3 → Projekt-Root bleibt das Repo-Verzeichnis.** Projekt wächst in diesem Verzeichnis. Importable Package heißt trotzdem `dvrrestorer` (sauberer Name für PyPI/GitHub später). `git init` im Projekt-Root als erster Schritt des Implementierungs-Plans.

Alle offenen Punkte sind damit aufgelöst.

# ASS Inpaint Mesh (`ass-inpaint-mesh.lua` + `rebuilder_worker.py`)

An Aegisub Automation 4 Lua macro paired with a CPU-friendly Python CLI worker that cleanly erases on-screen text over complex backgrounds (gradients, skies, walls, skin) and rebuilds the cleaned patch directly as native, playback-optimized ASS `\p1` vector drawing lines.

---

## How It Works

1. **Vector `\clip` Rasterization**: `rebuilder_worker.py` parses ASS `\clip(m ...)` commands (supporting `m`, `n`, `l`, `b` cubic Bezier curves, `s`/`p`/`c` B-splines, optional scale factors, and rectangular clips) and rasterizes them into a binary mask matching your video's resolution.
2. **Padded Bounding-Box CPU Inpainting**: The mask is dilated slightly (default `3px`) to cover anti-aliased text outlines, and only a padded crop around the `\clip` bounding box is processed—keeping 1080p/4K frame inpainting fast on CPU using OpenCV (`INPAINT_TELEA` / `INPAINT_NS`) or optional CPU-only `simple-lama-inpainting` (with instant OpenCV Navier-Stokes fallback if LaMa is unavailable).
3. **Greedy 2D Rectangle Merging (Anti-Lag Optimization)**: Instead of emitting thousands of tiny 1x1 or 4x4 pixel squares, adjacent grid cells within your chosen RGB Euclidean distance tolerance are greedily merged into larger rectangles (`m x1 y1 l x2 y1 x2 y2 x1 y2`) with averaged colors.
4. **Grouped `\p1` ASS Output**: All rectangles sharing the same quantized hex color (`\1c&HBBGGRR&`) are grouped into a single ASS Dialogue line (or minimal chunked lines) with `\an7\pos(0,0)\bord0\shad0\blur<val>\clip(...)\p1`. The `\clip(...)` keeps outer boundaries razor-sharp while `\blur` blends internal tile seams smoothly.

---

## Installation

### 1. Place the Aegisub Files

Copy **both** `ass-inpaint-mesh.lua` and `rebuilder_worker.py` into your Aegisub `automation/autoload/` directory so Aegisub discovers the macro automatically and the Lua script finds the Python worker alongside it (or place `rebuilder_worker.py` anywhere else and pick it once with the **Browse Python Script...** button):

- **Windows**:
  `%APPDATA%\Aegisub\automation\autoload\`
  (typically `C:\Users\<YourName>\AppData\Roaming\Aegisub\automation\autoload\`)
- **macOS**:
  `~/Library/Application Support/Aegisub/automation/autoload/`
- **Linux**:
  `~/.aegisub/automation/autoload/`

*(Then reload Automation scripts in Aegisub via **Automation → Automation... → Rescan Autoload Dir**, or restart Aegisub.)*

### 2. Install Python Dependencies

Ensure Python 3.8+ is installed and available on your system `PATH` as `python` (or `python3` / `py -3` on Windows).

#### Standard Setup (Fast CPU OpenCV Inpainting)

```bash
pip install numpy opencv-python Pillow
```

#### Full Setup (Including Optional CPU/GPU LaMa AI Inpainting)

If you want to use **LaMa AI (CPU - High Quality)** or **LaMa AI (GPU - High Quality)**, install the matching PyTorch build first, then install the remaining dependencies from `requirements.txt`:

```bash
# Option A: CPU-only PyTorch (smaller download for machines without a dedicated GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Option B: NVIDIA CUDA GPU PyTorch (recommended for GPU owners)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then install core packages + simple-lama-inpainting
pip install -r requirements.txt
```

> **Note:** On the first run with `LaMa AI`, `simple-lama-inpainting` downloads its pretrained LaMa model weights (`~200 MB`) once and caches them locally. If `simple-lama-inpainting` or `torch` is not installed, `rebuilder_worker.py` automatically falls back to OpenCV Navier-Stokes (`INPAINT_NS`).

*(Optional)* Having `ffmpeg` on `PATH` provides an automatic frame-extraction fallback if `cv2.VideoCapture` cannot decode an unusual container format.

---

## Aegisub User Workflow

1. Open your video and `.ass` subtitle script in Aegisub.
2. Create or select a dialogue line timed over the target scene and draw a **Vector Clip** (`\clip(m ...)`) tightly around the on-screen text you want to erase.
3. With that line selected in the subtitle grid, open **Automation → Gradient & Background Rebuilder (Inpaint Mesh)**.
4. Configure the dialog options:
   - **Inpainting Engine**:
     - `OpenCV Fast (Telea)` — Instant (~milliseconds); great for smooth gradients, skies, and thin text.
     - `OpenCV Smooth (Navier-Stokes)` — Better edge/texture continuation on walls and slightly textured surfaces.
     - `LaMa AI (CPU - High Quality)` — Deep-learning inpainting on CPU; best for complex textures or wider text blocks (falls back to Navier-Stokes if not installed).
     - `LaMa AI (GPU - High Quality)` — Same LaMa AI model accelerated on GPU (`CUDA` / Apple `MPS` / `DirectML`; automatically overrides `CPU` device to `GPU`).
   - **Compute Device**:
     - `CPU` — Forces CPU execution.
     - `GPU (CUDA / MPS)` — Runs LaMa AI on your GPU (`CUDA`, Apple `MPS`, or `DirectML`), falling back cleanly to CPU with a warning if no GPU backend is detected.
     - `Auto (Prefer GPU)` — Automatically uses GPU when available, otherwise runs on CPU silently. *(Note: OpenCV Telea and Navier-Stokes always run in milliseconds on CPU.)*
   - **Grid Resolution (Patch Size in px)** (`1`–`16`, default `4`):
     Controls tile granularity. `4` balances detail and speed; use `2`–`3` for steep gradients or `6`–`8` for flat backgrounds.
   - **Color Quantization / Merge Tolerance** (`0`–`30`, default `6`):
     Maximum Euclidean RGB distance for merging adjacent tiles into larger rectangles. Higher values produce far fewer ASS vertices and faster `libass` playback; `0` disables merging.
   - **Edge Feather / Blur (`\blur`)** (default `0.6`):
     Blurs internal rectangle seams naturally without blurring past the outer `\clip(...)` mask.
   - **Frame Range Mode**:
     - `Single Frame (Current Video Frame)` — Generates one static patch spanning the selected line's timing using the currently displayed video frame (ideal for locked shots).
     - `All Frames in Selected Line` — Rebuilds frame-by-frame across the line's start-to-end range (ideal for moving cameras or lighting changes).
   - **Python Executable** & **Python Script Path**:
     - Defaults to `python` and the auto-detected `rebuilder_worker.py`. Click **Browse Python Script...** to select `rebuilder_worker.py` via a file dialog.
     - All settings (including your chosen Python script path) are automatically loaded on launch from `?user/config/ass-inpaint-mesh.json` and saved when the dialog or script closes.
5. Click **Rebuild**. The macro comments out your original `\clip` guide line and inserts the optimized `\p1` vector patch lines immediately above it with clean sequential layers.

---

## Standalone CLI Usage (`rebuilder_worker.py`)

You can also invoke `rebuilder_worker.py` directly from the terminal for batch testing or custom pipelines:

```bash
# Process a single PNG frame and print raw ASS Dialogue lines
python rebuilder_worker.py \
  --image frame.png \
  --clip "\clip(m 400 200 l 800 200 800 280 400 280)" \
  --engine telea \
  --grid-size 4 \
  --tolerance 6 \
  --blur 0.6 \
  --format ass

# Process frames 100..110 from a video using a JSON config file
python rebuilder_worker.py --config job_request.json --output job_result.json
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--config` | Path to JSON configuration payload | — |
| `--video` | Source video path | — |
| `--image` | Pre-extracted PNG/image path | — |
| `--clip` | ASS `\clip(...)` drawing or coordinate string | Required |
| `--width`, `--height` | Video resolution (auto-detected if omitted) | Frame size |
| `--script-width`, `--script-height` | ASS `PlayResX` / `PlayResY` | Video size |
| `--engine` | `telea`, `ns`, or `lama` | `telea` |
| `--device` | `cpu`, `gpu`, `cuda`, `mps`, or `auto` | `cpu` |
| `--grid-size` | Mesh grid cell size in px (`1`–`16`) | `4` |
| `--tolerance` | Color quantization/merge tolerance (`0`–`30`) | `6` |
| `--blur` | ASS `\blur` value | `0.6` |
| `--dilate` | Mask dilation radius in px | `3` |
| `--frame` / `--frames` / `--start-frame` `--end-frame` | Target frame(s) | `0` |
| `--output` | Output file path (`-` or omit for `stdout`) | `stdout` |
| `--format` | `json` or `ass` | `json` |

---

## Moving Backgrounds: Google Colab GPU Notebook (`ass-inpaint-mesh-motion.ipynb`)

When erasing on-screen signs over **moving, shape-changing backgrounds** (drifting clouds, flowing water, camera pans, lighting transitions), single-frame spatial inpainting can flicker frame-to-frame. Use **`ass-inpaint-mesh-motion.ipynb`** on Google Colab (**G4 GPU**, L4 GPU, or T4 GPU) for temporally consistent video inpainting:

### When to Use Which Tool
- **Local (`ass-inpaint-mesh.lua` + `rebuilder_worker.py`)**: Instant CPU workflow directly inside Aegisub for static backgrounds or subtle shots.
- **Colab GPU (`ass-inpaint-mesh-motion.ipynb`)**: Auto-detects your Colab GPU tier (`G4 RTX PRO 6000 Blackwell 96GB`, `L4 24GB`, or `T4 16GB`) and unlocks higher-quality video inpainting engines:
  - **DiffuEraser + ProPainter Prior** *(G4 / L4 Recommended)*: Diffusion-based video inpainting (`Stable Diffusion 1.5 + BrushNet + Temporal Attention + PCM`) initialized by ProPainter optical flow—reconstructs fine moving textures inside large masks without blurring.
  - **ProPainter Full-Context + LaMa Spatial Refinement**: Combines ProPainter flow propagation with LaMa Fourier convolutions in deep mask centers plus optical-flow temporal smoothing.
  - **ProPainter AI (FP32 on G4 / FP16 on T4)**: Pure flow-guided transformer inpainting (ideal for clean anime gradients and sharp edges without diffusion synthesis).
  - **Optical-Flow Temporal Fallback**: Bidirectional Farneback flow + Navier-Stokes when CUDA/weights are unavailable.

### Colab Notebook Workflow
1. Upload `ass-inpaint-mesh-motion.ipynb` to Google Colab and select **Runtime → Change runtime type → `G4 GPU`** (or `L4 GPU` / `T4 GPU`).
2. **Cell 1 (Environment & GPU Setup)**: Auto-detects VRAM tier (`G4` unlocks FP32, `subvideo_length=120`, `neighbor_length=20`, `ref_stride=5`, and full `1920` crop size), installs dependencies, and sets up ProPainter + DiffuEraser + SimpleLama.
3. **Cell 2A & 2B (Video & `\clip` Input)**: Upload a trimmed `.mp4`/`.mkv` clip (or mount Google Drive), paste your Aegisub `\clip(...)` vector string, set the episode start timestamp offset (`HH:MM:SS.cs`), and choose **Option A: Static `\clip`** or **Option B: Motion-Tracked `\clip`** (Lucas-Kanade optical flow tracking).
4. **Cell 3 (Interactive Mask Preview)**: Previews Frame 0 with the dilated mask overlay and padded GPU crop box.
5. **Cell 4 (Temporal Video Inpainting)**: Runs the chosen GPU engine (or `Auto`), applies perimeter boundary color harmonization (`harmonize_boundary_colors`), and displays a side-by-side Original vs. Inpainted preview.
6. **Cell 5 & 6 (Vector Mesh Export)**: Applies Greedy 2D Rectangle Merging across every frame (or every `N`-th frame via `FRAME_STEP`) and downloads `rebuilt_motion_patch.ass` with full `[Script Info]` (`PlayResX`/`PlayResY`) and `[V4+ Styles]` headers ready for Aegisub.


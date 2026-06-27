# AGENTS.md — video-watermark

## Project

Python CLI that detects and removes watermarks from videos using temporal variance analysis + LaMa neural inpainting. Single script + one optional module (3 total `.py` files). No packaging, no tests, no linting, no CI.

## Commands

```bash
# Run (FFmpeg must be on PATH)
python dewatermark.py -i input.mp4 -o output.mp4

# With LaMa neural inpainting (better quality, slower)
python dewatermark.py -i input.mp4 -o output.mp4 --lama

# Apple Silicon GPU acceleration (MLX, sequential only)
python dewatermark.py -i input.mp4 -o output.mp4 --lama --mlx

# FLUX.1 Fill inpainting via mflux (MLX-native). WARNING: impractical for video —
# ~4 min/frame at 512² (needs >=512² or it produces noise; quantized fills are flat),
# and FLUX.1-Fill-dev is a gated ~35GB HF download. For watermark removal the default
# (OpenCV Telea) or --lama is both faster and higher quality. Kept as an optional backend.
python dewatermark.py -i input.mp4 -o output.mp4 --flux --flux-steps 20 --flux-res 512 --flux-quantize 8

# With a manual mask or region
python dewatermark.py -i input.mp4 -o output.mp4 -m mask.png
python dewatermark.py -i input.mp4 -o output.mp4 --region 100,50,200,80
```

No lint/formatter/test commands exist. If adding tooling, write a new `Makefile`.

## Architecture

| File | Purpose |
|------|---------|
| `dewatermark.py` | Sole entry point. CLI + pipeline: probe → detect → mask-prep → inpaint → re-encode. All I/O through tempdir. |
| `lama_mlx.py` | LaMa reimplementation for Apple MLX. `NHWC` tensor convention. One-time ONNX→MLX weight conversion to `~/.cache/dewatermark/lama_mlx.npz`. Untracked (not in git yet). |
| `requirements.txt` | Minimal deps. **Implicit extras** needed for various paths: `coremltools`, `mlx`, `onnx`, `mflux` (for `--flux` FLUX.1 Fill). These are imported lazily and not listed. |

## Pipeline gotchas

- **Full frame extraction to disk**: Every frame is extracted as JPEG to a tempdir, inpainted in-place, then re-encoded from disk. Memory is cheap but disk I/O is high.
- **Square crop constraint**: The mask crop box is forced square (LaMa expects 512×512). Padding around the watermark region can be larger than the 25% configured margin.
- **MLX is single-threaded**: Unlike the ONNX/CoreML paths (which use `multiprocessing.Pool`), the MLX path runs sequentially — no multiprocessing.
- **Automatic model download**: On first run, models are pulled from Hugging Face Hub to `~/.cache/dewatermark/`. The CoreML `.mlpackage` is not auto-downloaded — it must exist there already.
- **maxtasksperchild=500** on ONNX/CoreML pools to prevent memory leaks. ONNX pool cache is NOT cleared in the CoreML retry path (only CoreML cache is), which is a subtle bug if mixed backends are used.
- **Retry + fallback**: ONNX inpainting retries 3x on `RuntimeError` then falls back to OpenCV Telea. CoreML does the same.

## Commands worth teaching

- `python dewatermark.py --help` — lists every flag
- `ffprobe -v quiet -print_format json -show_format -show_streams <video>` — how the pipeline probes input
- Models live on Hugging Face under `Carve/LaMa-ONNX` (inpainting) and `qfisch/yolov8n-watermark-detection` (fallback detection)

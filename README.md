# Video Watermark Removal CLI

Automatically detect and remove watermarks from videos using AI inpainting.

## How it works

1. **Detection**: Temporal variance analysis identifies static overlay regions (watermarks). Falls back to YOLOv8n watermark detector if variance analysis fails.
2. **Inpainting**: LaMa (Large Mask Inpainting) fills in detected regions frame-by-frame.
3. **Encoding**: FFmpeg pipe-based I/O keeps memory usage constant regardless of video length.

## Requirements

- Python 3.9+
- FFmpeg installed and on PATH

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Automatic detection (recommended)
```bash
python dewatermark.py -i input.mp4 -o output.mp4
```

### Manual mask
```bash
python dewatermark.py -i input.mp4 -o output.mp4 -m mask.png
```

### Manual region (X,Y,W,H)
```bash
python dewatermark.py -i input.mp4 -o output.mp4 --region 50,20,200,60
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `-i` | required | Input video path |
| `-o` | required | Output video path |
| `-m` | — | Mask image (white = watermark) |
| `--region` | — | Bounding box as X,Y,W,H |
| `--model-dir` | `~/.cache/dewatermark` | Model download directory |
| `--feather` | 5 | Gaussian blur on mask edges (px) |
| `--crf` | 18 | Output quality (lower = better) |
| `--preset` | medium | FFmpeg encoding preset |
| `--sample-frames` | 30 | Frames for variance analysis |
| `--variance-threshold` | auto | Variance sensitivity (0-1, default: Otsu) |

## Models

Models are downloaded automatically on first run:

- **LaMa** ([Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)) — ~208MB, inpainting
- **YOLOv8n** ([qfisch/yolov8n-watermark-detection](https://huggingface.co/qfisch/yolov8n-watermark-detection)) — ~12MB, watermark detection (fallback only)

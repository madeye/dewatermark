#!/usr/bin/env python3
"""Automatic video watermark detection and removal using LaMa inpainting."""

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dewatermark")
LAMA_REPO = "Carve/LaMa-ONNX"
LAMA_FILE = "lama_fp32.onnx"
LAMA_COREML_PATH = os.path.join(DEFAULT_MODEL_DIR, "lama_fp16.mlpackage")
YOLO_REPO = "qfisch/yolov8n-watermark-detection"
YOLO_FILE = "yolov8n-watermark.onnx"


def parse_args():
    p = argparse.ArgumentParser(description="Remove watermarks from videos")
    p.add_argument("-i", "--input", required=True, help="Input video path")
    p.add_argument("-o", "--output", required=True, help="Output video path")
    p.add_argument("-m", "--mask", help="Mask image (white=watermark)")
    p.add_argument("--region", help="Watermark region as X,Y,W,H")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model cache dir")
    p.add_argument("--feather", type=int, default=5, help="Mask feather radius (px)")
    p.add_argument("--crf", type=int, default=18, help="Output CRF quality")
    p.add_argument("--preset", default="medium", help="FFmpeg encoding preset")
    p.add_argument("--sample-frames", type=int, default=30, help="Frames to sample for variance analysis")
    p.add_argument("--variance-threshold", type=float, default=None, help="Variance threshold (default: auto/otsu)")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers (default: CPU count)")
    p.add_argument("--lama", action="store_true", help="Use LaMa neural inpainting (slower, higher quality)")
    return p.parse_args()


def check_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("Error: ffmpeg and ffprobe must be installed and on PATH", file=sys.stderr)
        sys.exit(1)


def _ort_session(model_path):
    providers = []
    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(model_path, providers=providers)


def download_lama(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    return hf_hub_download(repo_id=LAMA_REPO, filename=LAMA_FILE, cache_dir=model_dir)


def load_lama(model_dir):
    path = download_lama(model_dir)
    return _ort_session(path)


def load_yolo(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILE, cache_dir=model_dir)
    print(f"YOLOv8n watermark detector loaded from {path}")
    return _ort_session(path)


def probe_video(path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    # Parse frame rate
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den if den else 30.0
    # Duration from format or stream
    duration = float(info["format"].get("duration", video_stream.get("duration", 0)))
    w = int(video_stream["width"])
    h = int(video_stream["height"])
    nb_frames = int(duration * fps) if duration else 0
    return {"width": w, "height": h, "fps": fps, "duration": duration, "nb_frames": nb_frames,
            "r_frame_rate": r_frame_rate}


def sample_frames(path, n, duration):
    """Extract N evenly-spaced frames from the video via FFmpeg seeks."""
    if duration <= 0:
        n = 1
    timestamps = np.linspace(0, max(duration - 0.1, 0), n)
    frames = []
    for ts in timestamps:
        cmd = [
            "ffmpeg", "-v", "quiet", "-ss", f"{ts:.3f}",
            "-i", path, "-frames:v", "1",
            "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        # Decode the raw frame — we need dimensions first
        # Re-probe is wasteful; instead decode via numpy
        raw = result.stdout
        if len(raw) == 0:
            continue
        # We'll decode properly after we know dimensions
        frames.append(raw)
    return frames


def _raw_to_frame(raw_bytes, w, h):
    expected = w * h * 3
    if len(raw_bytes) != expected:
        return None
    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))


def detect_watermark_variance(frames_raw, w, h, threshold=None):
    """Temporal variance analysis to find static watermark regions."""
    frames = []
    for raw in frames_raw:
        f = _raw_to_frame(raw, w, h)
        if f is not None:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames.append(gray)

    if len(frames) < 3:
        return None

    stack = np.stack(frames, axis=0)
    variance = np.var(stack, axis=0)

    # Normalize variance to 0-255
    var_max = variance.max()
    if var_max < 1e-6:
        return None
    var_norm = (variance / var_max * 255).astype(np.uint8)

    # Low variance = potential watermark → invert so watermark is bright
    inv = 255 - var_norm

    # Threshold
    if threshold is not None:
        _, binary = cv2.threshold(inv, int(threshold * 255), 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours, filter by size and position
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = w * h
    mask = np.zeros((h, w), dtype=np.uint8)
    found_any = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Skip tiny noise (< 0.05% of frame) or huge regions (> 15% of frame)
        if area < frame_area * 0.0005 or area > frame_area * 0.05:
            continue
        # Prefer edges/corners: check if centroid is in outer 30% of frame
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        margin_x = w * 0.3
        margin_y = h * 0.3
        in_edge = (cx < margin_x or cx > w - margin_x or
                   cy < margin_y or cy > h - margin_y)
        if not in_edge:
            continue
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        found_any = True

    return mask if found_any else None


def detect_watermark_yolo(session, frame, w, h, conf_thresh=0.25):
    """Run YOLOv8n watermark detector on a frame, return binary mask."""
    # Preprocess: resize to 640x640, normalize
    input_size = 640
    img = cv2.resize(frame, (input_size, input_size))
    img_float = img.astype(np.float32) / 255.0
    # HWC -> NCHW
    blob = np.transpose(img_float, (2, 0, 1))[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    preds = outputs[0]  # shape: (1, num_classes+4, num_detections) or (1, num_detections, num_classes+4)

    # YOLOv8 output: (1, 5, N) where rows are [x_center, y_center, w, h, conf]
    # or (1, N, 5) — handle both
    if preds.shape[1] < preds.shape[2]:
        preds = preds.transpose(0, 2, 1)  # -> (1, N, 5+)
    preds = preds[0]  # (N, 5+)

    # For single-class: columns are [cx, cy, bw, bh, score]
    # For multi-class: [cx, cy, bw, bh, class_scores...]
    if preds.shape[1] > 5:
        scores = preds[:, 4:].max(axis=1)
        boxes = preds[:, :4]
    else:
        scores = preds[:, 4]
        boxes = preds[:, :4]

    # Filter by confidence
    keep = scores > conf_thresh
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return None

    # NMS
    # Convert cx,cy,w,h to x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    indices = _nms(np.stack([x1, y1, x2, y2], axis=1), scores, iou_thresh=0.5)
    if len(indices) == 0:
        return None

    # Take highest confidence detection
    best = indices[np.argmax(scores[indices])]
    bx1 = max(0, int(x1[best] / input_size * w))
    by1 = max(0, int(y1[best] / input_size * h))
    bx2 = min(w, int(x2[best] / input_size * w))
    by2 = min(h, int(y2[best] / input_size * h))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[by1:by2, bx1:bx2] = 255
    print(f"YOLO detected watermark at ({bx1},{by1})-({bx2},{by2}) conf={scores[best]:.2f}")
    return mask


def _nms(boxes, scores, iou_thresh=0.5):
    """Simple NMS implementation."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


def prepare_mask(mask, w, h, feather):
    """Feather the mask and compute tight crop box with padding."""
    if feather > 0:
        ksize = feather * 2 + 1
        mask_float = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
        mask_float = np.clip(mask_float, 0, 255)
    else:
        mask_float = mask.astype(np.float32)

    # Find bounding box of non-zero mask region
    binary = (mask_float > 0).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return mask_float, (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(coords)

    # Add 25% padding
    pad_x = int(bw * 0.25)
    pad_y = int(bh * 0.25)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    # Make square (use larger dimension)
    cw = x2 - x1
    ch = y2 - y1
    side = max(cw, ch)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    # Adjust if clamped
    if x2 - x1 < side:
        x1 = max(0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0, y2 - side)

    crop_box = (x1, y1, x2, y2)
    return mask_float, crop_box


def _inpaint_frame_cv(args):
    """Worker: load frame, inpaint with OpenCV NS on full frame, save back."""
    frame_path, mask_u8 = args

    frame = cv2.imread(frame_path)
    if frame is None:
        return frame_path

    inpainted = cv2.inpaint(frame, mask_u8, inpaintRadius=12, flags=cv2.INPAINT_NS)
    cv2.imwrite(frame_path, inpainted)
    return frame_path


def _inpaint_frame_lama(args):
    """Worker: load frame, inpaint with LaMa (CoreML or ONNX), save back."""
    frame_path, lama_model_path, crop_box, mask_crop_norm, mask_input, alpha_crop, use_coreml = args
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    frame = cv2.imread(frame_path)
    if frame is None:
        return frame_path

    crop = frame[y1:y2, x1:x2].copy()
    crop_resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
    crop_input = crop_resized.astype(np.float32) / 255.0
    crop_nchw = np.transpose(crop_input, (2, 0, 1))[np.newaxis, ...]  # (1,3,512,512)

    if use_coreml:
        coreml_model = _get_worker_coreml(lama_model_path)
        pred = coreml_model.predict({"image": crop_nchw, "mask": mask_input})
        inpainted = list(pred.values())[0][0]  # (3,512,512)
    else:
        session = _get_worker_onnx(lama_model_path)
        input_name = session.get_inputs()[0].name
        mask_name = session.get_inputs()[1].name
        result = session.run(None, {input_name: crop_nchw, mask_name: mask_input})
        inpainted = result[0][0]  # (3,512,512)

    inpainted = np.transpose(inpainted, (1, 2, 0))  # (512,512,3)
    inpainted = np.clip(inpainted * 255, 0, 255).astype(np.uint8)
    inpainted_resized = cv2.resize(inpainted, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    blended = (inpainted_resized.astype(np.float32) * alpha_crop +
               crop.astype(np.float32) * (1 - alpha_crop))
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    cv2.imwrite(frame_path, frame)
    return frame_path


_worker_onnx_cache = {}
_worker_coreml_cache = {}


def _get_worker_onnx(model_path):
    """Per-process cached ONNX session."""
    pid = os.getpid()
    if pid not in _worker_onnx_cache:
        _worker_onnx_cache[pid] = _ort_session(model_path)
    return _worker_onnx_cache[pid]


def _get_worker_coreml(model_path):
    """Per-process cached CoreML model."""
    pid = os.getpid()
    if pid not in _worker_coreml_cache:
        import coremltools as ct
        _worker_coreml_cache[pid] = ct.models.MLModel(model_path)
    return _worker_coreml_cache[pid]


def process_video(input_path, output_path, mask_float, crop_box, info, crf, preset, workers,
                  lama_model_path=None, use_lama=False):
    """Extract frames → parallel inpaint → re-encode."""
    w, h = info["width"], info["height"]
    fps = info["r_frame_rate"]
    x1, y1, x2, y2 = crop_box

    # Precompute mask data
    mask_u8 = (mask_float > 0).astype(np.uint8) * 255
    mask_crop = mask_float[y1:y2, x1:x2]

    tmp_dir = tempfile.mkdtemp(prefix="dewatermark_")
    print(f"Temp directory: {tmp_dir}")

    try:
        # Step 1: Extract all frames
        print("Extracting frames...")
        extract_cmd = [
            "ffmpeg", "-v", "quiet", "-i", input_path,
            "-qscale:v", "2",
            os.path.join(tmp_dir, "%06d.jpg")
        ]
        subprocess.run(extract_cmd, check=True)

        frame_files = sorted(Path(tmp_dir).glob("*.jpg"))
        nb_frames = len(frame_files)
        print(f"Extracted {nb_frames} frames")

        if workers is None:
            workers = min(multiprocessing.cpu_count(), 8)

        # Step 2: Parallel inpaint
        if use_lama and lama_model_path:
            use_coreml = os.path.exists(LAMA_COREML_PATH)
            mask_crop_norm = mask_crop / 255.0
            mask_512 = cv2.resize(mask_crop_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
            mask_512_bin = (mask_512 > 0.1).astype(np.float32)
            mask_input = mask_512_bin[np.newaxis, np.newaxis, ...]
            alpha_crop = np.stack([mask_crop_norm] * 3, axis=-1)
            actual_model_path = LAMA_COREML_PATH if use_coreml else lama_model_path
            task_args = [
                (str(f), actual_model_path, crop_box, mask_crop_norm, mask_input, alpha_crop, use_coreml)
                for f in frame_files
            ]
            backend = "CoreML" if use_coreml else "ONNX"
            print(f"Inpainting with LaMa {backend} ({workers} workers)...")
            worker_fn = _inpaint_frame_lama
        else:
            task_args = [(str(f), mask_u8) for f in frame_files]
            print(f"Inpainting with OpenCV Telea ({workers} workers)...")
            worker_fn = _inpaint_frame_cv

        with multiprocessing.Pool(workers) as pool:
            for _ in tqdm(
                pool.imap(worker_fn, task_args),
                total=nb_frames, desc="Inpainting", unit="frame"
            ):
                pass

        # Step 3: Re-encode with audio from original
        print("Encoding output video...")
        encode_cmd = [
            "ffmpeg", "-v", "quiet", "-y",
            "-framerate", fps,
            "-i", os.path.join(tmp_dir, "%06d.jpg"),
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        subprocess.run(encode_cmd, check=True)

    finally:
        print("Cleaning up temp files...")
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    args = parse_args()
    check_ffmpeg()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print("Probing video...")
    info = probe_video(args.input)
    w, h = info["width"], info["height"]
    print(f"  {w}x{h} @ {info['fps']:.2f} fps, {info['duration']:.1f}s, ~{info['nb_frames']} frames")

    # Determine mask
    mask = None

    if args.mask:
        print(f"Loading manual mask from {args.mask}")
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: could not read mask image: {args.mask}", file=sys.stderr)
            sys.exit(1)
        mask = cv2.resize(mask, (w, h))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    elif args.region:
        print(f"Using manual region: {args.region}")
        parts = [int(x) for x in args.region.split(",")]
        if len(parts) != 4:
            print("Error: --region must be X,Y,W,H", file=sys.stderr)
            sys.exit(1)
        rx, ry, rw, rh = parts
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[ry:ry + rh, rx:rx + rw] = 255

    else:
        # Auto-detection pipeline
        print("Auto-detecting watermark...")

        # Step 1: Temporal variance analysis
        print("  Step 1: Temporal variance analysis...")
        frames_raw = sample_frames(args.input, args.sample_frames, info["duration"])
        print(f"  Sampled {len(frames_raw)} frames")
        mask = detect_watermark_variance(frames_raw, w, h, args.variance_threshold)

        if mask is not None:
            region_pct = np.count_nonzero(mask) / (w * h) * 100
            print(f"  Variance analysis found watermark region ({region_pct:.1f}% of frame)")
        else:
            # Step 2: YOLOv8n fallback
            print("  Variance analysis found nothing, trying YOLOv8n detector...")
            yolo_session = load_yolo(args.model_dir)
            # Use a middle frame for detection
            if frames_raw:
                mid_frame = _raw_to_frame(frames_raw[len(frames_raw) // 2], w, h)
                if mid_frame is not None:
                    mask = detect_watermark_yolo(yolo_session, mid_frame, w, h)

        if mask is None:
            print("Error: no watermark detected. Try providing a manual mask (-m) or region (--region).",
                  file=sys.stderr)
            sys.exit(1)

    lama_model_path = None
    if args.lama:
        print("Downloading LaMa inpainting model...")
        lama_model_path = download_lama(args.model_dir)
        print(f"LaMa model: {lama_model_path}")

    # Prepare mask and crop
    mask_float, crop_box = prepare_mask(mask, w, h, args.feather)
    x1, y1, x2, y2 = crop_box
    print(f"Crop region: ({x1},{y1})-({x2},{y2}) = {x2 - x1}x{y2 - y1}")

    # Process
    print("Processing video...")
    process_video(args.input, args.output, mask_float, crop_box, info,
                  args.crf, args.preset, args.workers,
                  lama_model_path=lama_model_path, use_lama=args.lama)
    print(f"Done! Output saved to {args.output}")


if __name__ == "__main__":
    main()

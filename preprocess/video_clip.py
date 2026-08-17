import argparse
import json
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TIMESTAMP_MARGIN_X = 6
TIMESTAMP_MARGIN_Y = 6
TIMESTAMP_STROKE_WIDTH = 2
MAX_FRAME_PIXELS = 128 * 28 * 28


def format_mmss(seconds: float) -> str:
    """Format seconds as mm:ss."""
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def draw_timestamp(image: Image.Image, timestamp_text: str) -> Image.Image:
    """Draw a timestamp in the top-right corner of the frame."""
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font_size = max(14, min(18, image.width // 30))
        font = ImageFont.truetype(TIMESTAMP_FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), timestamp_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    x = max(0, image.width - text_width - TIMESTAMP_MARGIN_X)
    y = max(0, TIMESTAMP_MARGIN_Y)

    draw.text(
        (x, y),
        timestamp_text,
        fill=(255, 255, 255),
        font=font,
        stroke_width=TIMESTAMP_STROKE_WIDTH,
        stroke_fill=(0, 0, 0),
    )
    return image


def resize_if_needed(image: Image.Image, max_pixels: int = MAX_FRAME_PIXELS) -> Image.Image:
    """Resize proportionally only when the frame exceeds the pixel budget."""
    image = image.convert("RGB")
    total_pixels = image.width * image.height
    if total_pixels <= max_pixels:
        return image

    resize_factor = (max_pixels / total_pixels) ** 0.5
    new_width = max(1, int(image.width * resize_factor))
    new_height = max(1, int(image.height * resize_factor))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def sample_frame_indices(total_frames: int, num_frames: int = 32) -> List[int]:
    """Uniformly sample frame indices over the full video timeline."""
    if total_frames <= 0:
        return []
    if total_frames <= num_frames:
        return list(range(total_frames))

    indices = np.linspace(0, total_frames - 1, num=num_frames)
    return [int(round(idx)) for idx in indices]


def extract_frames_with_timestamps(video_path: Path, out_root: Path, num_frames: int = 32) -> None:
    """Sample frames from a video, add timestamps, and save them by video id."""
    base = video_path.stem
    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 1.0

    frame_indices = sample_frame_indices(total_frames, num_frames=num_frames)
    if not frame_indices:
        print(f"[skip] no frames available: {video_path}")
        cap.release()
        return

    for save_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[warn] failed reading frame {frame_idx} from {video_path}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = resize_if_needed(Image.fromarray(frame_rgb))
        image = draw_timestamp(image, format_mmss(frame_idx / fps))
        image.save(out_dir / f"{base}_f{save_idx:02d}.png")

    cap.release()


def iter_unique_video_paths(dataset_path: Path) -> Iterable[str]:
    """Yield unique relative video paths from the public dataset JSON."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected the public dataset JSON to contain a list of videos.")

    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        video_path = item.get("video_path")
        if video_path and video_path not in seen:
            seen.add(video_path)
            yield video_path


def batch_extract_from_dataset(
    dataset_path: Path,
    video_root: Path,
    output_root: Path,
    num_frames: int = 32,
) -> None:
    """Extract frames for every unique video referenced by the dataset."""
    video_paths = list(iter_unique_video_paths(dataset_path))
    print(f"Found {len(video_paths)} unique videos in {dataset_path}")

    for idx, relative_path in enumerate(video_paths, start=1):
        video_path = Path(relative_path)
        if not video_path.is_absolute():
            video_path = video_root / video_path
        print(f"[{idx}/{len(video_paths)}] Processing {video_path} ...")
        extract_frames_with_timestamps(video_path, output_root, num_frames=num_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract timestamped frames from EG-VQA videos.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "train.json")
    parser.add_argument("--video-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "frames" / "train")
    parser.add_argument("--num-frames", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_extract_from_dataset(
        dataset_path=args.dataset,
        video_root=args.video_root,
        output_root=args.output_root,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()

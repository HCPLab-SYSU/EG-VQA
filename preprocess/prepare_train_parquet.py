import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ANNOTATIONS = PROJECT_ROOT / "data" / "train.json"
TRAIN_FRAMES_ROOT = PROJECT_ROOT / "data" / "frames" / "train"
TRAIN_PARQUET_DIR = PROJECT_ROOT / "data" / "train_set_parquet"
MAX_FRAMES = 32
SHARD_SIZE = 500


def iter_train_samples(annotation_path: Path) -> Iterable[Dict[str, Any]]:
    """Flatten the public video-level train JSON into QA-level samples."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected the public dataset JSON to contain a list of videos.")

    for video_item in data:
        if not isinstance(video_item, dict):
            continue

        video_id = video_item.get("video_id")
        questions = video_item.get("questions") or []
        metadata = video_item.get("metadata") or {}
        if not video_id or not isinstance(questions, list):
            continue

        for question_item in questions:
            if not isinstance(question_item, dict):
                continue
            yield {
                "video_id": video_id,
                "video_path": video_item.get("video_path", ""),
                "duration": video_item.get("duration"),
                "data_source": video_item.get("data_source"),
                "metadata": metadata,
                "question_id": question_item.get("question_id"),
                "question": question_item.get("question", ""),
                "type": question_item.get("type"),
                "answer": question_item.get("answer", ""),
                "evidence": question_item.get("evidence") or [],
            }


def natural_sort_key(path: Path) -> List[Any]:
    """Sort frame names numerically, e.g. f2 before f10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_image_bytes(frame_dir: Path, max_frames: int = 32) -> List[bytes]:
    """Read up to max_frames PNG/JPEG files from one extracted-frame directory."""
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = sorted(
        [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in image_suffixes],
        key=natural_sort_key,
    )[:max_frames]
    return [path.read_bytes() for path in image_paths]


def build_ground_truth(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the public annotation schema in the Parquet ground truth."""
    metadata = dict(sample.get("metadata") or {})

    return {
        "video_id": sample.get("video_id"),
        "question_id": sample.get("question_id"),
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "type": sample.get("type"),
        "data_source": sample.get("data_source"),
        "duration": sample.get("duration"),
        "evidence": sample.get("evidence", []),
        "metadata": metadata,
    }


def load_training_rows(annotation_path: Path, frames_root: Path, max_frames: int = 32) -> List[Dict[str, Any]]:
    """Build multimodal training rows from annotations and extracted frames."""
    rows = []
    missing_frame_dirs = 0

    for sample in iter_train_samples(annotation_path):
        video_id = sample["video_id"]
        frame_dir = frames_root / video_id
        if not frame_dir.is_dir():
            missing_frame_dirs += 1
            continue

        image_bytes = load_image_bytes(frame_dir, max_frames=max_frames)
        if not image_bytes:
            continue

        rows.append(
            {
                "images": image_bytes,
                "problem": "<image>" * len(image_bytes) + sample["question"],
                "answer": json.dumps(build_ground_truth(sample), ensure_ascii=False),
            }
        )

    if missing_frame_dirs:
        print(f"[warn] skipped {missing_frame_dirs} samples because their frame directories were missing")
    return rows


def write_parquet_shards(rows: List[Dict[str, Any]], output_dir: Path, shard_size: int = 500) -> None:
    """Write training rows as Parquet shards."""
    if not rows:
        raise ValueError("No training rows were built. Check annotations and extracted frames.")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx, start in enumerate(range(0, len(rows), shard_size)):
        shard = rows[start : start + shard_size]
        table = pa.Table.from_pydict(
            {
                "images": [row["images"] for row in shard],
                "problem": [row["problem"] for row in shard],
                "answer": [row["answer"] for row in shard],
            }
        )
        shard_path = output_dir / f"part-{shard_idx:05d}.parquet"
        pq.write_table(table, shard_path)
        print(f"[{shard_idx + 1}] wrote {len(shard)} samples -> {shard_path}")


def main() -> None:
    rows = load_training_rows(TRAIN_ANNOTATIONS, TRAIN_FRAMES_ROOT, max_frames=MAX_FRAMES)
    print(f"Built {len(rows)} training samples")
    write_parquet_shards(rows, TRAIN_PARQUET_DIR, shard_size=SHARD_SIZE)


if __name__ == "__main__":
    main()

"""Run inference for an EG-Reasoner checkpoint on the public EG-VQA test split.

The script intentionally keeps model paths, frame paths, and output names outside
the source code. Each run is identified by ``--run-name`` and stored as one JSON
file under the selected output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROMPT_TEMPLATE = """You are a video understanding assistant. Given a video and a question, respond in the following format:
First output the seen relevant video segments within <evidence> </evidence> tags in chronological order. 
Then based on the observed video and evidence, analyze and reason through the question. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags.

Format: <evidence>Time:MM:SS-MM:SS, Des: description (one or more lines)</evidence>. <think> reasoning based on evidence </think>. <answer> concise direct answer </answer>.

IMPORTANT RULES:
- Output EXACTLY 3 sections: <evidence>, <think>, <answer>
- Evidence: Maximum 5 lines (NOT more than 5). Each line MUST follow: "Time:MM:SS-MM:SS, Des: ..."
- NO repeated descriptions or time segments
- Think: 2-4 sentences maximum, based ONLY on the evidence above
- Answer: 1 short sentence, no explanations

## GOOD Example (CORRECT):
Question: What tool does the person use?
Response:
<evidence>
Time:00:10-00:15, Des: Person picks up a hammer from the table.
Time:00:16-00:22, Des: He uses the hammer to hit the nail.
</evidence>

<think>
The person picks up a hammer and then uses it to hit a nail, clearly showing the tool being used.
</think>

<answer>
The person uses a hammer.
</answer>

## BAD Example (WRONG - DO NOT DO THIS):
<evidence>
Time:00:00-00:05, Des: Person does something.
Time:00:05-00:10, Des: Person does something.
Time:00:10-00:15, Des: Person does something.
[repeating many times...]
Time:04:55-05:00, Des: Person does something.
</evidence>
[missing think and answer sections]

Now answer the following question:
Question: {question}
"""


def natural_sort_key(path: Path) -> List[Any]:
    """Sort frame names numerically, for example f2 before f10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_test_samples(annotation_path: Path) -> List[Dict[str, Any]]:
    """Flatten the public video-level test JSON into question-level samples."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected the public test JSON to contain a list of videos.")

    samples: List[Dict[str, Any]] = []
    for video_item in data:
        if not isinstance(video_item, dict):
            continue

        video_id = video_item.get("video_id")
        questions = video_item.get("questions") or []
        if not video_id or not isinstance(questions, list):
            continue

        for question_item in questions:
            if not isinstance(question_item, dict):
                continue
            question_id = question_item.get("question_id")
            if not question_id:
                continue
            samples.append(
                {
                    "video_id": video_id,
                    "question_id": question_id,
                    "video_path": video_item.get("video_path", ""),
                    "duration": video_item.get("duration"),
                    "data_source": video_item.get("data_source"),
                    "metadata": video_item.get("metadata") or {},
                    "question": question_item.get("question", ""),
                    "type": question_item.get("type"),
                    "answer": question_item.get("answer", ""),
                    "evidence": question_item.get("evidence") or [],
                }
            )
    return samples


def resolve_model_path(args: argparse.Namespace) -> Tuple[Path, Optional[str]]:
    """Resolve a direct model path or the latest veRL checkpoint."""
    if args.model_path is not None:
        return args.model_path, None

    checkpoint_dir = args.checkpoint_root / args.project_name / args.experiment_name
    tracker_path = checkpoint_dir / "latest_global_step.txt"
    if tracker_path.is_file():
        step_token = tracker_path.read_text(encoding="utf-8").strip()
    else:
        step_dirs = list(checkpoint_dir.glob("global_step_*"))
        step_values = [
            int(path.name.removeprefix("global_step_"))
            for path in step_dirs
            if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
        ]
        if not step_values:
            raise FileNotFoundError(
                f"No veRL checkpoints found in {checkpoint_dir}. "
                "Run training and model merging first."
            )
        step_token = str(max(step_values))

    if not step_token.isdigit():
        raise ValueError(f"Invalid latest checkpoint value in {tracker_path}: {step_token!r}")

    model_path = (
        checkpoint_dir
        / f"global_step_{int(step_token)}"
        / "actor"
        / "huggingface"
    )
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Merged actor model not found: {model_path}. "
            "Run EG-Reasoner/scripts/model_merger.py on the actor directory first."
        )

    run_name = f"{args.project_name}_{args.experiment_name}_step_{int(step_token)}"
    return model_path, run_name


def frame_paths_for_video(frames_root: Path, video_id: str, max_frames: int) -> List[Path]:
    """Return extracted frame paths for one video."""
    frame_dir = frames_root / video_id
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    paths = sorted(
        [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key,
    )[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No frame images found in: {frame_dir}")
    return paths


def extract_answer(text: str) -> Optional[str]:
    """Extract the answer section, including responses with an open tag only."""
    match = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"<answer>(.*?)$", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_model_response(text: str) -> Dict[str, Optional[str]]:
    """Extract evidence, reasoning, and answer from a model response."""
    text = text or ""
    result: Dict[str, Optional[str]] = {"evidence": None, "think": None, "answer": None}

    evidence_match = re.search(r"<evidence>(.*?)</evidence>", text, re.DOTALL | re.IGNORECASE)
    if evidence_match:
        result["evidence"] = evidence_match.group(1).strip()
    else:
        evidence_match = re.search(r"<evidence>(.*?)(?=<think>|<answer>|$)", text, re.DOTALL | re.IGNORECASE)
        if evidence_match:
            result["evidence"] = evidence_match.group(1).strip()

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        result["think"] = think_match.group(1).strip()
    else:
        think_match = re.search(r"<think>(.*?)(?=<answer>|$)", text, re.DOTALL | re.IGNORECASE)
        if think_match:
            result["think"] = think_match.group(1).strip()

    result["answer"] = extract_answer(text)
    return result


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON through a temporary file so interrupted runs can resume."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


class VisionInputBuilder:
    """Prepare and cache visual inputs for one video at a time."""

    def __init__(self, processor, process_vision_info, frames_root: Path, max_frames: int, max_pixels: int):
        self.processor = processor
        self.process_vision_info = process_vision_info
        self.frames_root = frames_root
        self.max_frames = max_frames
        self.max_pixels = max_pixels
        self.cached_frame_dir: Optional[Path] = None
        self.cached_multimodal_data: Optional[Dict[str, Any]] = None
        self.cached_processor_kwargs: Optional[Dict[str, Any]] = None

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        import numpy as np

        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def build(self, video_id: str, question: str) -> Dict[str, Any]:
        paths = frame_paths_for_video(self.frames_root, video_id, self.max_frames)
        frame_dir = paths[0].parent

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(path), "max_pixels": self.max_pixels}
                    for path in paths
                ]
                + [{"type": "text", "text": PROMPT_TEMPLATE.format(question=question)}],
            }
        ]

        if frame_dir != self.cached_frame_dir:
            image_inputs, video_inputs, video_kwargs = self.process_vision_info(
                messages,
                return_video_kwargs=True,
            )
            multimodal_data: Dict[str, Any] = {}
            if image_inputs is not None:
                multimodal_data["image"] = [self._to_numpy(image) for image in image_inputs]
            if video_inputs is not None:
                multimodal_data["video"] = [self._to_numpy(video) for video in video_inputs]

            processor_kwargs: Dict[str, Any] = {}
            for key, value in (video_kwargs or {}).items():
                processor_kwargs[key] = value[0] if isinstance(value, (list, tuple)) and value else value

            self.cached_frame_dir = frame_dir
            self.cached_multimodal_data = multimodal_data
            self.cached_processor_kwargs = processor_kwargs

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": text,
            "multi_modal_data": self.cached_multimodal_data or {},
            "mm_processor_kwargs": self.cached_processor_kwargs or {},
        }


def build_result(sample: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    """Combine public ground truth fields with inference output fields."""
    result = {
        "video_id": sample["video_id"],
        "question_id": sample["question_id"],
        "video_path": sample["video_path"],
        "duration": sample["duration"],
        "data_source": sample["data_source"],
        "metadata": sample["metadata"],
        "question": sample["question"],
        "type": sample["type"],
        "answer": sample["answer"],
        "evidence": sample["evidence"],
    }
    result.update(fields)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EG-Reasoner inference on EG-VQA test annotations.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Direct path or Hugging Face model ID. If omitted, use the latest veRL checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "EG-Reasoner" / "checkpoints",
        help="Root directory containing project and experiment checkpoint folders.",
    )
    parser.add_argument("--project-name", type=str, default="EG-VQA")
    parser.add_argument("--experiment-name", type=str, default="EG-Reasoner")
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "data" / "test.json")
    parser.add_argument("--frames-root", type=Path, default=PROJECT_ROOT / "data" / "frames" / "test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used by vLLM tensor parallelism; default: 1.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Comma-separated GPU IDs visible to vLLM, for example 0,1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be a positive integer.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")
    if args.max_frames < 1:
        raise ValueError("--max-frames must be a positive integer.")

    model_path, checkpoint_run_name = resolve_model_path(args)

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ.setdefault("QWEN_VL_VIDEO_READER_BACKEND", "decord")

    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    samples = load_test_samples(args.annotations)
    if not samples:
        raise ValueError(f"No question samples found in {args.annotations}")

    run_name = args.run_name or checkpoint_run_name or re.sub(r"[^A-Za-z0-9_.-]+", "_", model_path.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{run_name}.json"
    partial_path = args.output_dir / f"{run_name}.partial.json"

    previous_results: List[Dict[str, Any]] = []
    if not args.overwrite:
        resume_path = partial_path if partial_path.exists() else output_path if output_path.exists() else None
        if resume_path is not None:
            previous_results = json.loads(resume_path.read_text(encoding="utf-8"))
            previous_results = [
                {key: value for key, value in result.items() if key != "processing_time"}
                for result in previous_results
                if isinstance(result, dict)
            ]
    processed_ids = {result.get("question_id") for result in previous_results}
    pending_samples = [sample for sample in samples if sample["question_id"] not in processed_ids]

    logger = logging.getLogger("eg_vqa_evaluation")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Loaded %d questions; %d pending", len(samples), len(pending_samples))

    if not pending_samples:
        atomic_write_json(output_path, previous_results)
        logger.info("All questions are already present in %s", output_path)
        return

    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_frames},
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    input_builder = VisionInputBuilder(
        processor,
        process_vision_info,
        args.frames_root,
        args.max_frames,
        args.max_pixels,
    )

    results = list(previous_results)
    for batch_start in tqdm(range(0, len(pending_samples), args.batch_size), desc="Evaluating"):
        batch = pending_samples[batch_start : batch_start + args.batch_size]
        inputs = []
        prepared_samples = []

        for sample in batch:
            try:
                inputs.append(input_builder.build(sample["video_id"], sample["question"]))
                prepared_samples.append(sample)
            except Exception as exc:
                results.append(build_result(sample, error=f"input preparation failed: {exc}"))

        if inputs:
            try:
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                for sample, output in zip(prepared_samples, outputs):
                    raw_text = output.outputs[0].text if output.outputs else ""
                    results.append(
                        build_result(
                            sample,
                            model_response_raw=raw_text,
                            parsed_response=parse_model_response(raw_text),
                        )
                    )
            except Exception as exc:
                for sample in prepared_samples:
                    results.append(build_result(sample, error=f"inference failed: {exc}"))

        atomic_write_json(partial_path, results)

    atomic_write_json(output_path, results)
    if partial_path.exists():
        partial_path.unlink()
    logger.info("Saved %d results to %s", len(results), output_path)


if __name__ == "__main__":
    main()

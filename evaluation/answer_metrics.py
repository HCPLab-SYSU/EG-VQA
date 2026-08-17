"""Compute relaxed and strict answer accuracy from evaluate.py results."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THREAD_LOCAL = threading.local()


def create_evaluation_prompt(
    metadata: str,
    question: str,
    golden_answer: str,
    model_answer: str,
    title: Optional[str] = None,
) -> str:
    """Build the answer-judge prompt used by the original relax-score script."""
    if title:
        prompt = """You are an expert specializing in evaluating whether a respondent's answer after watching a video matches the golden answer. We will provide the video's title, video segments' descriptions, question, golden answer, and the response to be judged below.

### Video's Title:
{title}

### Video Segments' Descriptions (in chronological order):
{metadata}

### Question: 
{question}

### Golden Answer: 
{golden_answer}

### Response to be judged: 
{model_answer}

### Rules:
1. If the response to be judged contains ALL key information of the golden answer or expresses the same meaning using other sentences or synonyms, it is considered a match, and the output is 1.
2. If the response to be judged does NOT contain the key information from the golden answer, it is considered a mismatch, and the output is 0.
3. The response to be judged should NOT contain any content that is contradictory, conflicting, or unreasonable when inferred from the video content description. If such content exists, it is considered a mismatch, and the output is 0.
4. If the response to be judged contains MOST of the key information of the golden answer, and does NOT contain any information that is contradictory, conflicting, or unreasonable when inferred from the video content description, it is considered a partial match, and the output is 0.5.

### Instructions:
Follow the format below and do not give any extra outputs:
Answer: 0 (if the response does not match)
Answer: 0.5 (if the response partially matches)
Answer: 1 (if the response matches)
"""
        return prompt.format(
            title=title,
            metadata=metadata,
            question=question,
            golden_answer=golden_answer,
            model_answer=model_answer,
        )

    prompt = """You are an expert specializing in evaluating whether a respondent's answer after watching a video matches the golden answer. We will provide the video's video segments' descriptions, question, golden answer, and the response to be judged below.

### Video Segments' Descriptions (in chronological order):
{metadata}

### Question: 
{question}

### Golden Answer: 
{golden_answer}

### Response to be judged: 
{model_answer}

### Rules:
1. If the response to be judged contains ALL key information of the golden answer or expresses the same meaning using other sentences or synonyms, it is considered a match, and the output is 1.
2. If the response to be judged does NOT contain the key information from the golden answer, it is considered a mismatch, and the output is 0.
3. The response to be judged should NOT contain any content that is contradictory, conflicting, or unreasonable when inferred from the video content description. If such content exists, it is considered a mismatch, and the output is 0.
4. If the response to be judged contains MOST of the key information of the golden answer, and does NOT contain any information that is contradictory, conflicting, or unreasonable when inferred from the video content description, it is considered a partial match, and the output is 0.5.

### Instructions:
Follow the format below and do not give any extra outputs:
Answer: 0 (if the response does not match)
Answer: 0.5 (if the response partially matches)
Answer: 1 (if the response matches)
"""
    return prompt.format(
        metadata=metadata,
        question=question,
        golden_answer=golden_answer,
        model_answer=model_answer,
    )


def extract_score(response: str) -> Optional[float]:
    """Extract exactly 0, 0.5, or 1 from the judge response."""
    match = re.search(r"(?im)^\s*Answer\s*:\s*(0(?:\.5)?|1(?:\.0)?)\b", response or "")
    if not match:
        return None
    value = float(match.group(1))
    return 1.0 if value == 1.0 else 0.5 if value == 0.5 else 0.0


def load_evaluation_items(path: Path) -> List[Dict[str, Any]]:
    """Load the question-level list written by evaluate.py."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError("Expected evaluate.py output to be a list of question results.")
    return [item for item in data if isinstance(item, dict)]


def _metadata_fields(item: Dict[str, Any]) -> tuple[Optional[str], List[str]]:
    metadata = item.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None, []

    title = metadata.get("title")
    segments = metadata.get("segments")
    if not isinstance(segments, list):
        segments = metadata.get("descriptions", [])

    descriptions: List[str] = []
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict):
                description = segment.get("description", segment.get("descriptions", ""))
            else:
                description = str(segment)
            if str(description).strip():
                descriptions.append(str(description).strip())
    return (str(title) if title else None), descriptions


def prepare_samples(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map the evaluate.py schema to the answer-judge schema."""
    samples: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        parsed = item.get("parsed_response") or {}
        if not isinstance(parsed, dict):
            parsed = {}

        title, descriptions = _metadata_fields(item)
        if title is None and item.get("title"):
            title = str(item["title"])
        if not descriptions and isinstance(item.get("video_segments"), list):
            descriptions = [str(text).strip() for text in item["video_segments"] if str(text).strip()]
        question_id = item.get("question_id") or f"{item.get('video_id', 'sample')}_{index}"
        question_type = item.get("type") or item.get("question_type") or "unknown"
        model_answer = parsed.get("answer") or item.get("model_answer") or ""
        ground_truth = item.get("answer") or item.get("standard_answer") or ""

        samples.append(
            {
                "video_id": item.get("video_id"),
                "question_id": question_id,
                "data_source": item.get("data_source"),
                "question": item.get("question", ""),
                "question_type": question_type,
                "title": title,
                "video_segments": descriptions,
                "standard_answer": str(ground_truth),
                "model_answer": str(model_answer),
                "model_response_raw": item.get("model_response_raw"),
                "source_error": item.get("error"),
            }
        )
    return samples


def get_client(api_key: str, base_url: Optional[str]):
    """Create one OpenAI-compatible client per worker thread."""
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        _THREAD_LOCAL.client = client
    return client


def evaluate_sample(
    sample: Dict[str, Any],
    api_key: str,
    base_url: Optional[str],
    model: str,
    delay: float,
) -> Dict[str, Any]:
    """Score one answer and keep failures in the denominator."""
    result = {
        "video_id": sample["video_id"],
        "question_id": sample["question_id"],
        "data_source": sample["data_source"],
        "question": sample["question"],
        "question_type": sample["question_type"],
        "standard_answer": sample["standard_answer"],
        "model_answer": sample["model_answer"],
        "evaluation_score": 0.0,
        "strict_score": 0.0,
        "evaluation_response": None,
    }

    if sample["source_error"]:
        result["error"] = f"evaluate.py error: {sample['source_error']}"
        return result
    if not sample["standard_answer"]:
        result["error"] = "missing ground-truth answer"
        return result
    if not sample["model_answer"].strip():
        result["error"] = "missing model answer"
        return result

    metadata = "\n".join(f"- {text}" for text in sample["video_segments"])
    prompt = create_evaluation_prompt(
        metadata=metadata,
        question=sample["question"],
        golden_answer=sample["standard_answer"],
        model_answer=sample["model_answer"],
        title=sample["title"],
    )

    try:
        response = get_client(api_key, base_url).chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        response_text = response.choices[0].message.content or ""
        score = extract_score(response_text)
        if score is None:
            result["error"] = "could not parse judge score"
        else:
            result["evaluation_score"] = score
            result["strict_score"] = 1.0 if score == 1.0 else 0.0
        result["evaluation_response"] = response_text
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if delay > 0:
            time.sleep(delay)
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate relaxed and strict accuracy overall and by question type."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result.get("question_type") or "unknown", []).append(result)

    def stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(items)
        score_sum = sum(float(item.get("evaluation_score", 0.0)) for item in items)
        strict_sum = sum(float(item.get("strict_score", 0.0)) for item in items)
        return {
            "total_samples": total,
            "relaxed_score_sum": round(score_sum, 4),
            "relaxed_accuracy": round(score_sum / total, 4) if total else 0.0,
            "strict_correct": int(strict_sum),
            "strict_accuracy": round(strict_sum / total, 4) if total else 0.0,
            "failed_samples": sum(1 for item in items if item.get("error")),
        }

    return {
        "overall_stats": stats(results),
        "type_stats": {question_type: stats(items) for question_type, items in sorted(grouped.items())},
    }


def atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute relaxed and strict answer accuracy.")
    parser.add_argument("--input", required=True, type=Path, help="JSON output from evaluation/evaluate.py.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--api-model", default=os.getenv("OPENAI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--api-key", "--api_key", dest="api_key", default=None)
    parser.add_argument(
        "--api-base",
        "--api_base",
        "--base-url",
        dest="base_url",
        default=os.getenv("OPENAI_BASE_URL"),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.save_interval < 1:
        raise ValueError("--workers and --save-interval must be positive.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = args.output or args.input.with_name(f"{args.input.stem}_answer_metrics.json")
    partial_path = output_path.with_suffix(".partial.json")
    samples = prepare_samples(load_evaluation_items(args.input))

    previous: List[Dict[str, Any]] = []
    if not args.overwrite:
        resume_path = partial_path if partial_path.exists() else output_path if output_path.exists() else None
        if resume_path:
            saved = json.loads(resume_path.read_text(encoding="utf-8"))
            previous = saved.get("samples", []) if isinstance(saved, dict) else []

    processed = {item.get("question_id") for item in previous}
    pending = [sample for sample in samples if sample["question_id"] not in processed]
    results = list(previous)
    needs_api = any(
        not sample["source_error"] and sample["standard_answer"] and sample["model_answer"].strip()
        for sample in pending
    )
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if needs_api and not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before answer scoring.")

    print(f"Total samples: {len(samples)}; pending: {len(pending)}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_sample,
                sample,
                api_key or "",
                args.base_url,
                args.api_model,
                args.delay,
            ): sample
            for sample in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % args.save_interval == 0:
                atomic_write(partial_path, {**summarize(results), "samples": results})
                print(f"Saved checkpoint: {len(results)}/{len(samples)}")

    sample_order = {sample["question_id"]: index for index, sample in enumerate(samples)}
    results.sort(key=lambda item: sample_order.get(item.get("question_id"), len(samples)))
    payload = {
        "source_file": str(args.input),
        "judge_model": args.api_model,
        **summarize(results),
        "samples": results,
    }
    atomic_write(output_path, payload)
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(payload["overall_stats"], ensure_ascii=False, indent=2))
    print(f"Saved answer metrics to: {output_path}")


if __name__ == "__main__":
    main()

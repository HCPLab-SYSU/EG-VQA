"""Compute IoU-F1 and EG-F1 from the JSON written by evaluate.py.

The evaluator consumes the public EG-VQA schema directly.  In particular,
ground-truth evidence is read from ``evidence`` and model evidence is read
from ``parsed_response.evidence``.  Legacy result files with
``original_evidence`` are accepted as a small compatibility convenience.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EG_THRESHOLDS = "0.3:0.5,0.3:0.75,0.5:0.75"
DEFAULT_IOU_THRESHOLDS = "0.1,0.3,0.5,0.7"


@dataclass
class Evidence:
    start: float
    end: float
    description: str
    is_valid: bool = True
    parse_error: str = ""


def parse_number(value: Any) -> Optional[float]:
    """Parse seconds, MM:SS, or HH:MM:SS into seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return float(text)
        match = re.fullmatch(r"(\d+):(\d{1,2}):(\d+(?:\.\d+)?)", text)
        if match:
            hours, minutes, seconds = match.groups()
            if int(minutes) >= 60 or float(seconds) >= 60:
                return None
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        match = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", text)
        if match:
            minutes, seconds = match.groups()
            if float(seconds) >= 60:
                return None
            return int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None
    return None


def invalid_evidence(description: str, error: str) -> Evidence:
    return Evidence(0.0, 0.0, description, is_valid=False, parse_error=error)


def parse_timestamp_pair(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    start = parse_number(value[0])
    end = parse_number(value[1])
    if start is None or end is None:
        return None
    return start, end


def parse_evidence(value: Any) -> List[Evidence]:
    """Parse either public annotation evidence or model evidence text."""
    if isinstance(value, list):
        parsed: List[Evidence] = []
        for item in value:
            if not isinstance(item, dict):
                parsed.append(invalid_evidence("", "evidence item is not an object"))
                continue
            timestamps = item.get("timestamp", item.get("timestamps"))
            description = item.get("description", item.get("descriptions", ""))
            description = str(description or "").strip()
            pair = parse_timestamp_pair(timestamps)
            if pair is None:
                parsed.append(invalid_evidence(description, f"invalid timestamp: {timestamps!r}"))
                continue
            start, end = pair
            if start >= end:
                parsed.append(invalid_evidence(description, f"start must be before end: {timestamps!r}"))
            else:
                parsed.append(Evidence(start, end, description))
        return parsed

    if not isinstance(value, str) or not value.strip():
        return []

    parsed = []
    pattern = r"Time:\s*([^,]+?)\s*,\s*Des:\s*(.*?)(?=\n\s*Time:|$)"
    for time_part, description in re.findall(pattern, value, flags=re.IGNORECASE | re.DOTALL):
        description = re.sub(r"\s+", " ", description).strip()
        if "-" not in time_part:
            parsed.append(invalid_evidence(description, f"missing '-' in time range: {time_part!r}"))
            continue
        start_text, end_text = (part.strip() for part in time_part.split("-", 1))
        start = parse_number(start_text)
        end = parse_number(end_text)
        if start is None or end is None:
            parsed.append(invalid_evidence(description, f"invalid time range: {time_part!r}"))
        elif start >= end:
            parsed.append(invalid_evidence(description, f"start must be before end: {time_part!r}"))
        elif not description:
            parsed.append(invalid_evidence(description, "description is empty"))
        else:
            parsed.append(Evidence(start, end, description))
    return parsed


def load_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError("Expected a list of question-level evaluation results.")
    return [item for item in data if isinstance(item, dict)]


def ground_truth_evidence(item: Dict[str, Any]) -> List[Evidence]:
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        return parse_evidence(evidence)
    return parse_evidence(item.get("original_evidence", []))


def predicted_evidence(item: Dict[str, Any]) -> List[Evidence]:
    parsed = item.get("parsed_response")
    if isinstance(parsed, dict) and parsed.get("evidence") is not None:
        return parse_evidence(parsed.get("evidence"))
    if item.get("model_evidence") is not None:
        return parse_evidence(item.get("model_evidence"))
    # Compatibility with the old output where top-level evidence was the
    # model's string and original_evidence held the ground truth.
    if isinstance(item.get("evidence"), str):
        return parse_evidence(item.get("evidence"))
    return []


def compute_iou(gt: Evidence, pred: Evidence) -> float:
    intersection = max(0.0, min(gt.end, pred.end) - max(gt.start, pred.start))
    union = max(gt.end, pred.end) - min(gt.start, pred.start)
    return intersection / union if union > 0 else 0.0


def match_by_score(score_matrix: np.ndarray) -> List[Tuple[int, int]]:
    if score_matrix.size == 0:
        return []
    rows, cols = linear_sum_assignment(-score_matrix)
    return list(zip(rows.tolist(), cols.tolist()))


def score_counts(
    ground_truths: Sequence[Evidence],
    predictions: Sequence[Evidence],
    matches: Iterable[Tuple[int, int]],
    is_valid: Any,
) -> Dict[str, Any]:
    valid_predictions = sum(1 for prediction in predictions if prediction.is_valid)
    valid_matches: List[Dict[str, Any]] = []
    for gt_index, pred_index in matches:
        gt = ground_truths[gt_index]
        pred = predictions[pred_index]
        if pred.is_valid and is_valid(gt_index, pred_index):
            valid_matches.append(
                {
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "iou": compute_iou(gt, pred),
                }
            )

    n_gt = len(ground_truths)
    n_pred = len(predictions)
    valid_count = len(valid_matches)
    precision = valid_count / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = valid_count / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "valid_matches": valid_count,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_pred_valid": valid_predictions,
        "n_pred_invalid": n_pred - valid_predictions,
        "matches": valid_matches,
    }


def aggregate(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(name: str) -> float:
        return float(np.mean([item[name] for item in per_sample])) if per_sample else 0.0

    total_gt = sum(item["n_gt"] for item in per_sample)
    total_pred = sum(item["n_pred"] for item in per_sample)
    total_valid = sum(item["valid_matches"] for item in per_sample)
    weighted_precision = total_valid / total_pred if total_pred else (1.0 if total_gt == 0 else 0.0)
    weighted_recall = total_valid / total_gt if total_gt else (1.0 if total_pred == 0 else 0.0)
    weighted_f1 = (
        2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        if weighted_precision + weighted_recall
        else 0.0
    )
    return {
        "num_samples": len(per_sample),
        "avg_f1": mean("f1"),
        "avg_precision": mean("precision"),
        "avg_recall": mean("recall"),
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "total_ground_truths": total_gt,
        "total_predictions": total_pred,
        "total_valid_matches": total_valid,
        "total_invalid_predictions": sum(item["n_pred_invalid"] for item in per_sample),
    }


def iou_matrix(ground_truths: Sequence[Evidence], predictions: Sequence[Evidence]) -> np.ndarray:
    matrix = np.zeros((len(ground_truths), len(predictions)), dtype=np.float32)
    for gt_index, gt in enumerate(ground_truths):
        for pred_index, pred in enumerate(predictions):
            if pred.is_valid:
                matrix[gt_index, pred_index] = compute_iou(gt, pred)
    return matrix


def evaluate_iou(
    items: Sequence[Dict[str, Any]], thresholds: Sequence[float]
) -> Dict[str, Any]:
    per_threshold: Dict[str, List[Dict[str, Any]]] = {format_threshold(t): [] for t in thresholds}
    per_sample: List[Dict[str, Any]] = []

    for index, item in enumerate(items):
        ground_truths = ground_truth_evidence(item)
        predictions = predicted_evidence(item)
        ious = iou_matrix(ground_truths, predictions)
        matches = match_by_score(ious)
        sample = {
            "video_id": item.get("video_id", f"sample_{index}"),
            "question_id": item.get("question_id"),
            "question": item.get("question", ""),
            "n_gt": len(ground_truths),
            "n_pred": len(predictions),
            "n_pred_valid": sum(prediction.is_valid for prediction in predictions),
            "thresholds": {},
        }
        for threshold in thresholds:
            counts = score_counts(
                ground_truths,
                predictions,
                matches,
                lambda gt_index, pred_index, threshold=threshold: (
                    ious[gt_index, pred_index] >= threshold
                ),
            )
            per_threshold[format_threshold(threshold)].append(counts)
            sample["thresholds"][format_threshold(threshold)] = counts
        per_sample.append(sample)

    return {
        "thresholds": [float(t) for t in thresholds],
        "metrics_by_threshold": {
            key: aggregate(values) for key, values in per_threshold.items()
        },
        "per_sample_results": per_sample,
    }


class SemanticEncoder:
    """Lazy BGE-M3 loader used only when EG-F1 is requested."""

    def __init__(self, model_dir: Path, device: str) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_dir = model_dir
        if device.lower().startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA is unavailable; using CPU for BGE-M3.")
        self.device = device
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if not (self.model_dir / "modules.json").is_file():
            from huggingface_hub import snapshot_download

            print(f"Downloading BGE-M3 to {self.model_dir} ...")
            snapshot_download(repo_id="BAAI/bge-m3", local_dir=str(self.model_dir))
        print(f"Loading BGE-M3 from {self.model_dir} on {self.device} ...")
        self.model = SentenceTransformer(str(self.model_dir), device=self.device)

    def encode_texts(self, texts: Iterable[str]) -> Dict[str, np.ndarray]:
        unique_texts = list(dict.fromkeys(text for text in texts if text))
        if not unique_texts:
            return {}
        print(f"Encoding {len(unique_texts)} unique evidence descriptions ...")
        embeddings = self.model.encode(
            unique_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return {
            text: np.asarray(embedding, dtype=np.float32)
            for text, embedding in zip(unique_texts, embeddings)
        }

    def similarity_matrix(
        self,
        ground_truths: Sequence[Evidence],
        predictions: Sequence[Evidence],
        embedding_cache: Dict[str, np.ndarray],
    ) -> np.ndarray:
        matrix = np.zeros((len(ground_truths), len(predictions)), dtype=np.float32)
        for gt_index, gt in enumerate(ground_truths):
            gt_embedding = embedding_cache.get(gt.description)
            if gt_embedding is None:
                continue
            for pred_index, pred in enumerate(predictions):
                if not pred.is_valid:
                    continue
                pred_embedding = embedding_cache.get(pred.description)
                if pred_embedding is not None:
                    matrix[gt_index, pred_index] = float(
                        np.clip(np.dot(gt_embedding, pred_embedding), 0.0, 1.0)
                    )
        return matrix


def evaluate_eg(
    items: Sequence[Dict[str, Any]],
    thresholds: Sequence[Tuple[float, float]],
    encoder: SemanticEncoder,
) -> Dict[str, Any]:
    per_threshold: Dict[str, List[Dict[str, Any]]] = {
        threshold_key(iou, sim): [] for iou, sim in thresholds
    }
    parsed_samples: List[Tuple[Dict[str, Any], List[Evidence], List[Evidence]]] = []
    all_descriptions: List[str] = []
    for item in items:
        ground_truths = ground_truth_evidence(item)
        predictions = predicted_evidence(item)
        parsed_samples.append((item, ground_truths, predictions))
        all_descriptions.extend(
            evidence.description
            for evidence in [*ground_truths, *predictions]
            if evidence.description
        )
    embedding_cache = encoder.encode_texts(all_descriptions)
    per_sample: List[Dict[str, Any]] = []

    for index, (item, ground_truths, predictions) in enumerate(parsed_samples):
        ious = iou_matrix(ground_truths, predictions)
        similarities = encoder.similarity_matrix(ground_truths, predictions, embedding_cache)
        sample = {
            "video_id": item.get("video_id", f"sample_{index}"),
            "question_id": item.get("question_id"),
            "question": item.get("question", ""),
            "n_gt": len(ground_truths),
            "n_pred": len(predictions),
            "n_pred_valid": sum(prediction.is_valid for prediction in predictions),
            "thresholds": {},
        }
        for iou_threshold, sim_threshold in thresholds:
            key = threshold_key(iou_threshold, sim_threshold)
            valid_matrix = np.where(
                (ious >= iou_threshold) & (similarities >= sim_threshold),
                ious * similarities,
                0.0,
            )
            matches = match_by_score(valid_matrix)
            counts = score_counts(
                ground_truths,
                predictions,
                matches,
                lambda gt_index, pred_index, iou_threshold=iou_threshold, sim_threshold=sim_threshold: (
                    ious[gt_index, pred_index] >= iou_threshold
                    and similarities[gt_index, pred_index] >= sim_threshold
                ),
            )
            per_threshold[key].append(counts)
            sample["thresholds"][key] = counts
        per_sample.append(sample)

    return {
        "thresholds": [
            {"iou": float(iou), "similarity": float(sim)} for iou, sim in thresholds
        ],
        "metrics_by_threshold": {
            key: aggregate(values) for key, values in per_threshold.items()
        },
        "per_sample_results": per_sample,
    }


def format_threshold(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def threshold_key(iou: float, sim: float) -> str:
    return f"iou@{format_threshold(iou)}_sim@{format_threshold(sim)}"


def parse_float_list(text: str) -> List[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value < 0 or value > 1 for value in values):
        raise ValueError("Thresholds must be comma-separated numbers in [0, 1].")
    return values


def parse_threshold_pairs(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for token in text.split(","):
        if not token.strip():
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError("EG-F1 thresholds must use the format iou:similarity,iou:similarity.")
        iou, sim = (float(part.strip()) for part in parts)
        if not 0 <= iou <= 1 or not 0 <= sim <= 1:
            raise ValueError("EG-F1 thresholds must be in [0, 1].")
        pairs.append((iou, sim))
    if not pairs:
        raise ValueError("At least one EG-F1 threshold pair is required.")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute IoU-F1 and EG-F1 evidence metrics.")
    parser.add_argument("--input", required=True, type=Path, help="JSON output from evaluation/evaluate.py.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--eg-thresholds",
        default=DEFAULT_EG_THRESHOLDS,
        help=f"EG-F1 pairs as iou:similarity,... (default: {DEFAULT_EG_THRESHOLDS}).",
    )
    parser.add_argument(
        "--iou-thresholds",
        default=DEFAULT_IOU_THRESHOLDS,
        help=f"IoU-F1 thresholds as comma-separated values (default: {DEFAULT_IOU_THRESHOLDS}).",
    )
    parser.add_argument(
        "--semantic-model-dir",
        type=Path,
        default=PROJECT_ROOT / "EG-Reasoner" / "Science_Bert",
        help="Local directory for BGE-M3; downloaded there when absent.",
    )
    parser.add_argument("--semantic-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    output_path = args.output or args.input.with_name(f"{args.input.stem}_evidence_metrics.json")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    items = load_items(args.input)
    eg_thresholds = parse_threshold_pairs(args.eg_thresholds)
    iou_thresholds = parse_float_list(args.iou_thresholds)
    print(f"Loaded {len(items)} samples from {args.input}")
    encoder = SemanticEncoder(args.semantic_model_dir, args.semantic_device)
    result = {
        "source_file": str(args.input),
        "eg_f1": evaluate_eg(items, eg_thresholds, encoder),
        "iou_f1": evaluate_iou(items, iou_thresholds),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("EG-F1:")
    for key, metrics in result["eg_f1"]["metrics_by_threshold"].items():
        print(f"  {key}: {metrics['avg_f1']:.4f}")
    print("IoU-F1:")
    for key, metrics in result["iou_f1"]["metrics_by_threshold"].items():
        print(f"  IoU@{key}: {metrics['avg_f1']:.4f}")
    print(f"Saved evidence metrics to: {output_path}")


if __name__ == "__main__":
    main()

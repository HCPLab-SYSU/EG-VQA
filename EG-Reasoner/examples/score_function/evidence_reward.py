import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# Standard line format expected inside the <evidence> block.
# Example: Time: 00:19-00:28, Des: ...
EVIDENCE_LINE_PATTERN = re.compile(r"Time:\s*([^,]+?)\s*,\s*Des:\s*(.*?)(?=\n\s*Time:|$)", re.DOTALL | re.IGNORECASE)


_HES_EVALUATOR = None
_HES_EVALUATOR_LOCK = threading.Lock()


def safe_json_loads(value: str) -> Any:
    """Safely parse a JSON string, returning None on failure."""
    try:
        return json.loads(value)
    except Exception:
        return None


def extract_tag_content(text: str, tag: str) -> str:
    """Extract content from a tag like <tag>...</tag>."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_answer(model_response: str) -> Tuple[str, str]:
    """
    提取 <answer> 内容，并尽量修补缺失的闭合标签。

    返回:
    - answer_text: 提取到的答案，提取失败时返回空字符串
    - normalized_response: 可能补过 </answer> 的响应文本
    """
    if not model_response:
        return "", "" if model_response is None else str(model_response)

    # 先尝试匹配完整的 <answer>...</answer>
    answer_pattern = re.search(r"<answer>(.*?)</answer>", model_response, re.DOTALL | re.IGNORECASE)
    if answer_pattern:
        return answer_pattern.group(1).strip(), model_response

    # 如果只有开标签没有闭合标签，则把结尾视为 answer 内容，并补上 </answer>
    answer_pattern_open = re.search(r"<answer>(.*?)$", model_response, re.DOTALL | re.IGNORECASE)
    if answer_pattern_open:
        answer = answer_pattern_open.group(1).strip()
        return answer, model_response + "\n</answer>"

    return "", model_response


def parse_model_response(response_text: str) -> Tuple[Dict[str, str], str]:
    """
    解析模型输出，尽量从不完整标签中恢复 evidence / think / answer。

    设计原则：
    - format_reward 仍然看原始输出，格式错就给 0
    - evidence_reward / accuracy_reward 用这里修补后的结果取内容
    - 只修补“轻微标签缺失”场景，避免把完全错误的输出硬凑成合法格式
    """
    result: Dict[str, str] = {"evidence": "", "think": "", "answer": ""}
    modified_response = "" if response_text is None else str(response_text)

    # ----- 提取 evidence -----
    evidence_match = re.search(r"<evidence>(.*?)</evidence>", modified_response, re.DOTALL | re.IGNORECASE)
    if evidence_match:
        result["evidence"] = evidence_match.group(1).strip()
    else:
        # 如果 evidence 缺失闭合标签，就在下一个标签前截断，避免污染 think / answer
        ev_start = re.search(r"<evidence>(.*?)$", modified_response, re.DOTALL | re.IGNORECASE)
        if ev_start:
            content = ev_start.group(1)
            next_tag_match = re.search(r"(<think>|<answer>)", content, re.DOTALL | re.IGNORECASE)
            if next_tag_match:
                end_pos = next_tag_match.start()
                result["evidence"] = content[:end_pos].strip()
                modified_response = (
                    modified_response[: ev_start.start() + len("<evidence>") + end_pos]
                    + "</evidence>\n\n"
                    + modified_response[ev_start.start() + len("<evidence>") + end_pos :]
                )
            else:
                result["evidence"] = content.strip()
                modified_response = modified_response + "\n</evidence>"

    # ----- 提取 think -----
    think_match = re.search(r"<think>(.*?)</think>", modified_response, re.DOTALL | re.IGNORECASE)
    if think_match:
        result["think"] = think_match.group(1).strip()
    else:
        # 如果 think 缺失闭合标签，就在 <answer> 前截断，避免污染 answer
        th_start = re.search(r"<think>(.*?)$", modified_response, re.DOTALL | re.IGNORECASE)
        if th_start:
            content = th_start.group(1)
            answer_tag_match = re.search(r"<answer>", content, re.DOTALL | re.IGNORECASE)
            if answer_tag_match:
                end_pos = answer_tag_match.start()
                result["think"] = content[:end_pos].strip()
                modified_response = (
                    modified_response[: th_start.start() + len("<think>") + end_pos]
                    + "</think>\n\n"
                    + modified_response[th_start.start() + len("<think>") + end_pos :]
                )
            else:
                result["think"] = content.strip()
                modified_response = modified_response + "\n</think>"

    # ----- 提取 answer -----
    answer, updated_response = extract_answer(modified_response)
    result["answer"] = answer
    if updated_response != modified_response:
        modified_response = updated_response

    return result, modified_response


def parse_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    """
    Normalize training-time ground_truth into a structured dictionary.

    Supported inputs:
    - plain answer string
    - JSON / dict structure containing question, answer, evidence, and metadata
    """
    payload = ground_truth
    if isinstance(payload, str):
        maybe_json = safe_json_loads(payload)
        payload = maybe_json if isinstance(maybe_json, dict) else payload

    if isinstance(payload, dict):
        answer = payload.get("answer") or payload.get("golden_answer") or payload.get("solution") or ""
        evidence = payload.get("evidence")
        if evidence is None:
            # Backward compatibility for earlier intermediate files.
            evidence = payload.get("original_evidence") or []
        question = payload.get("question") or payload.get("problem") or ""
        metadata = payload.get("metadata")
        if metadata is None:
            # Backward compatibility for earlier intermediate files.
            metadata = payload.get("meta_info") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        title = payload.get("title")
        video_segments = payload.get("video_segments")

        if title is None:
            title = metadata.get("title")
        if video_segments is None:
            segments = metadata.get("segments")
            if isinstance(segments, list):
                video_segments = [
                    segment.get("description", "") if isinstance(segment, dict) else str(segment)
                    for segment in segments
                ]
            else:
                # Backward compatibility for the old metadata.descriptions list.
                old_descriptions = metadata.get("descriptions", [])
                video_segments = old_descriptions if isinstance(old_descriptions, list) else []

        return {
            "answer": "" if answer is None else str(answer),
            "evidence": evidence if evidence is not None else [],
            "question": question,
            "metadata": metadata,
            "title": title,
            "video_segments": video_segments if video_segments is not None else [],
            "duration": payload.get("duration"),
        }

    return {
        "answer": "" if payload is None else str(payload),
        "evidence": [],
        "question": "",
        "metadata": {},
        "title": None,
        "video_segments": [],
        "duration": None,
    }


@dataclass
class Evidence:
    """Structured evidence item."""

    start: float
    end: float
    description: str
    is_valid: bool = True
    parse_error: str = ""


class BGEM3Similarity:
    """
    Text similarity wrapper for BGE-M3.

    This is lazily initialized so we do not load the semantic model until
    evidence reward is actually used.
    """

    def __init__(self, models_dir: str = "./Science_Bert", device: str = "cuda") -> None:
        self.model_name = "BAAI/bge-m3"
        self.models_dir = Path(models_dir)
        self.device = self._resolve_device(device)
        self.local_model_dir = self._prepare_local_model()
        self.model = self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve device, falling back to CPU when CUDA is unavailable."""
        if device.lower() == "cpu":
            return "cpu"

        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                return device
        except Exception:
            pass

        print("[device check] CUDA unavailable, fallback to CPU.")
        return "cpu"

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def _find_weight_files(self, local_model_dir: Path) -> List[Path]:
        """Recursively locate model weight files."""
        possible_weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        ]

        found_files: List[Path] = []
        for filename in possible_weight_files:
            found_files.extend(list(local_model_dir.rglob(filename)))
        return found_files

    def _is_model_already_downloaded(self, local_model_dir: Path) -> bool:
        """Check whether a complete local model already exists."""
        if not local_model_dir.exists() or not local_model_dir.is_dir():
            return False

        required_meta_files = ["modules.json"]
        possible_support_files = [
            "config.json",
            "tokenizer_config.json",
            "sentence_bert_config.json",
        ]

        has_required_meta = all((local_model_dir / f).exists() for f in required_meta_files)
        has_support_file = any((local_model_dir / f).exists() for f in possible_support_files)
        has_weight_file = len(self._find_weight_files(local_model_dir)) > 0
        return has_required_meta and has_support_file and has_weight_file

    def _remove_incomplete_model_dir(self, local_model_dir: Path) -> None:
        """Remove incomplete model directories before redownloading."""
        if local_model_dir.exists() and local_model_dir.is_dir():
            import shutil

            print(f"[model check] remove incomplete model dir: {local_model_dir}")
            shutil.rmtree(local_model_dir, ignore_errors=True)

    def _download_model_to_local(self, local_model_dir: Path) -> Path:
        """Download the model into a fixed local directory."""
        from huggingface_hub import snapshot_download

        self._ensure_dir(local_model_dir)
        snapshot_download(
            repo_id=self.model_name,
            local_dir=str(local_model_dir),
            ignore_patterns=[
                "*.DS_Store",
                "**/.DS_Store",
                "imgs/*",
                "images/*",
                "*.png",
                "*.jpg",
                "*.jpeg",
                "*.gif",
                "*.svg",
                "*.md",
            ],
        )
        return local_model_dir

    def _prepare_local_model(self) -> Path:
        """Ensure a complete local semantic model exists."""
        local_model_dir = self.models_dir
        self._ensure_dir(self.models_dir)

        if local_model_dir.exists() and local_model_dir.is_dir():
            if self._is_model_already_downloaded(local_model_dir):
                print(f"[model check] use local semantic model: {local_model_dir}")
                return local_model_dir
            self._remove_incomplete_model_dir(local_model_dir)

        print(f"[model check] download semantic model to: {local_model_dir}")
        local_model_dir = self._download_model_to_local(local_model_dir)
        print(f"[model check] semantic model ready: {local_model_dir}")
        return local_model_dir

    def _load_model(self):
        """Load the BGE-M3 model."""
        from sentence_transformers import SentenceTransformer

        print(f"[model load] loading semantic model: {self.local_model_dir}, device={self.device}")
        return SentenceTransformer(str(self.local_model_dir), device=self.device)

    def similarity(self, text1: str, text2: str) -> float:
        """Return normalized cosine similarity for two texts."""
        text1 = "" if text1 is None else str(text1).strip()
        text2 = "" if text2 is None else str(text2).strip()

        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        embeddings = self.model.encode(
            [text1, text2],
            batch_size=2,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        score = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, score))

    def __call__(self, text1: str, text2: str) -> float:
        return self.similarity(text1, text2)


class HESEvaluator:
    """Holistic Evidence Score evaluator."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        sim_threshold: float = 0.75,
        semantic_model_dir: str = "./Science_Bert",
        semantic_device: str = "cuda",
    ):
        self.iou_threshold = iou_threshold
        self.sim_threshold = sim_threshold
        self.semantic_model_dir = semantic_model_dir
        self.semantic_device = semantic_device
        self.semantic_model: Optional[BGEM3Similarity] = None
        self._semantic_model_lock = threading.Lock()

    def _get_semantic_model(self) -> BGEM3Similarity:
        """Lazy-init the semantic model to reduce startup cost."""
        if self.semantic_model is None:
            with self._semantic_model_lock:
                if self.semantic_model is None:
                    print("\ninitializing semantic similarity model...")
                    self.semantic_model = BGEM3Similarity(
                        models_dir=self.semantic_model_dir,
                        device=self.semantic_device,
                    )
                    print("semantic similarity model ready\n")
        return self.semantic_model

    def time_to_seconds(self, time_str: str) -> Tuple[float, bool]:
        """Convert a time string to seconds."""
        if not time_str:
            return 0.0, False

        time_str = str(time_str).strip()
        if re.match(r"^\d+(?:\.\d+)?$", time_str):
            try:
                return float(time_str), True
            except ValueError:
                return 0.0, False

        match = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\.(\d+))?$", time_str)
        if not match:
            return 0.0, False

        groups = match.groups()
        if groups[2] is not None:
            hours = int(groups[0])
            minutes = int(groups[1])
            seconds = int(groups[2])
            frac_seconds = float(f"0.{groups[3] or '0'}")
            if hours > 23 or minutes > 59 or seconds > 59:
                return 0.0, False
            return hours * 3600 + minutes * 60 + seconds + frac_seconds, True

        minutes = int(groups[0])
        seconds = int(groups[1])
        frac_seconds = float(f"0.{groups[3] or '0'}")
        if minutes > 59 or seconds > 59:
            return 0.0, False
        return minutes * 60 + seconds + frac_seconds, True

    def parse_evidence_from_json(self, evidence_data: Any) -> List[Evidence]:
        """Parse evidence from either a model string or a dataset list."""
        if isinstance(evidence_data, str):
            return self._parse_evidence_string(evidence_data)

        if not isinstance(evidence_data, list):
            return []

        evidence_list = []
        for item in evidence_data:
            if not isinstance(item, dict):
                continue

            timestamps = item.get("timestamp")
            if timestamps is None:
                # Backward compatibility for earlier intermediate files.
                timestamps = item.get("timestamps", [])

            description = item.get("description")
            if description is None:
                # Backward compatibility for earlier intermediate files.
                description = item.get("descriptions", "")

            if len(timestamps) < 2:
                continue
            try:
                start = (
                    float(timestamps[0])
                    if isinstance(timestamps[0], (int, float))
                    else self.time_to_seconds(timestamps[0])[0]
                )
                end = (
                    float(timestamps[1])
                    if isinstance(timestamps[1], (int, float))
                    else self.time_to_seconds(timestamps[1])[0]
                )
                evidence_list.append(Evidence(start, end, description, is_valid=True))
            except (ValueError, TypeError):
                evidence_list.append(
                    Evidence(0, 0, description, is_valid=False, parse_error=f"time parse failed: {timestamps}")
                )
        return evidence_list

    def _parse_evidence_string(self, evidence_str: str) -> List[Evidence]:
        """Parse the model-generated <evidence> text block."""
        evidence_list = []
        if not evidence_str or not isinstance(evidence_str, str):
            return evidence_list

        matches = re.findall(EVIDENCE_LINE_PATTERN, evidence_str)
        for time_part, description in matches:
            description = re.sub(r"\s+", " ", description).strip()
            if "-" not in time_part:
                evidence_list.append(
                    Evidence(0, 0, description, is_valid=False, parse_error=f"missing '-' in time span: {time_part}")
                )
                continue

            start_str, end_str = time_part.split("-", 1)
            start_seconds, start_valid = self.time_to_seconds(start_str.strip())
            end_seconds, end_valid = self.time_to_seconds(end_str.strip())

            if not start_valid or not end_valid:
                evidence_list.append(
                    Evidence(0, 0, description, is_valid=False, parse_error=f"invalid time span: {time_part}")
                )
                continue
            if start_seconds >= end_seconds:
                evidence_list.append(
                    Evidence(0, 0, description, is_valid=False, parse_error=f"start >= end: {time_part}")
                )
                continue

            evidence_list.append(Evidence(start_seconds, end_seconds, description, is_valid=True))
        return evidence_list

    @staticmethod
    def compute_iou(gt: Evidence, pred: Evidence) -> float:
        """Compute temporal IoU between two evidence spans."""
        intersection_start = max(gt.start, pred.start)
        intersection_end = min(gt.end, pred.end)
        intersection = max(0.0, intersection_end - intersection_start)
        union_start = min(gt.start, pred.start)
        union_end = max(gt.end, pred.end)
        union = union_end - union_start
        if union == 0:
            return 0.0
        return intersection / union

    def compute_semantic_similarity(self, gt: Evidence, pred: Evidence) -> float:
        """Compute semantic similarity between evidence descriptions."""
        return self._get_semantic_model()(gt.description, pred.description)

    def hungarian_match(
        self, ground_truths: List[Evidence], predictions: List[Evidence]
    ) -> List[Tuple[int, int, float, float, float]]:
        """Match ground-truth and predicted evidence with Hungarian matching."""
        valid_predictions = [(idx, pred) for idx, pred in enumerate(predictions) if pred.is_valid]
        if not ground_truths or not valid_predictions:
            return []

        m = len(ground_truths)
        n = len(valid_predictions)
        iou_matrix = np.zeros((m, n), dtype=np.float32)
        sim_matrix = np.zeros((m, n), dtype=np.float32)
        score_matrix = np.zeros((m, n), dtype=np.float32)
        for i, gt in enumerate(ground_truths):
            for j, (_, pred) in enumerate(valid_predictions):
                iou = self.compute_iou(gt, pred)
                sim = self.compute_semantic_similarity(gt, pred)
                iou_matrix[i, j] = iou
                sim_matrix[i, j] = sim
                score_matrix[i, j] = 0.5 * iou + 0.5 * sim

        size = max(m, n)
        cost_matrix = np.full((size, size), 1.0, dtype=np.float32)
        cost_matrix[:m, :n] = -score_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for i, j in zip(row_indices, col_indices):
            if i < m and j < n:
                matches.append(
                    (
                        i,
                        valid_predictions[j][0],
                        float(score_matrix[i, j]),
                        float(iou_matrix[i, j]),
                        float(sim_matrix[i, j]),
                    )
                )
        return matches

    def evaluate_sample(
        self,
        ground_truths: List[Evidence],
        predictions: List[Evidence],
        use_soft_matching: bool = True,
    ) -> Dict[str, float]:
        """Evaluate one sample with the full or hard-only EG-F1 variant."""
        if not ground_truths and not predictions:
            return {"hes": 1.0, "precision": 1.0, "recall": 1.0}
        if not ground_truths or not predictions:
            return {"hes": 0.0, "precision": 0.0, "recall": 0.0}

        matches = self.hungarian_match(ground_truths, predictions)
        valid_matches = 0
        matched_score_sum = 0.0
        for gt_idx, pred_idx, score, iou, sim in matches:
            matched_score_sum += score
            is_valid = iou >= self.iou_threshold and sim >= self.sim_threshold
            if is_valid:
                valid_matches += 1

        n_gt = len(ground_truths)
        n_pred = len(predictions)
        hard_precision = valid_matches / n_pred if n_pred else 0.0
        hard_recall = valid_matches / n_gt if n_gt else 0.0
        hard_hes = (
            0.0 if hard_precision + hard_recall == 0 else 2 * hard_precision * hard_recall / (hard_precision + hard_recall)
        )

        soft_precision = matched_score_sum / n_pred if n_pred else 0.0
        soft_recall = matched_score_sum / n_gt if n_gt else 0.0
        soft_hes = (
            0.0 if soft_precision + soft_recall == 0 else 2 * soft_precision * soft_recall / (soft_precision + soft_recall)
        )

        hes = 0.5 * soft_hes + 0.5 * hard_hes if use_soft_matching else hard_hes
        return {"hes": hes, "precision": hard_precision, "recall": hard_recall}


def get_hes_evaluator(
    iou_threshold: float,
    sim_threshold: float,
    semantic_model_dir: str,
    semantic_device: str,
) -> HESEvaluator:
    """Cache the evaluator so training does not reinitialize it each step."""
    global _HES_EVALUATOR
    if _HES_EVALUATOR is None:
        with _HES_EVALUATOR_LOCK:
            if _HES_EVALUATOR is None:
                _HES_EVALUATOR = HESEvaluator(
                    iou_threshold=iou_threshold,
                    sim_threshold=sim_threshold,
                    semantic_model_dir=semantic_model_dir,
                    semantic_device=semantic_device,
                )
    return _HES_EVALUATOR


def evidence_reward(
    predict_str: str,
    ground_truth: Any,
    iou_threshold: float = 0.5,
    sim_threshold: float = 0.75,
    semantic_model_dir: str = "./Science_Bert",
    semantic_device: str = "cuda",
    use_soft_matching: bool = True,
) -> float:
    """Compute reward for the <evidence> block."""
    parsed_ground_truth = parse_ground_truth(ground_truth)
    gt_evidence = parsed_ground_truth["evidence"]
    if not gt_evidence:
        return 0.0

    # evidence reward 使用修补后的解析结果，尽量挽回轻微标签错误带来的损失
    parsed_response, _ = parse_model_response(predict_str)
    predicted_evidence_text = parsed_response["evidence"]
    evaluator = get_hes_evaluator(
        iou_threshold=iou_threshold,
        sim_threshold=sim_threshold,
        semantic_model_dir=semantic_model_dir,
        semantic_device=semantic_device,
    )
    ground_truths = evaluator.parse_evidence_from_json(gt_evidence)
    predictions = evaluator.parse_evidence_from_json(predicted_evidence_text)
    return float(
        evaluator.evaluate_sample(
            ground_truths,
            predictions,
            use_soft_matching=use_soft_matching,
        )["hes"]
    )

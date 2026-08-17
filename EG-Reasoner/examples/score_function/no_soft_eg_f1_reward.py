"""EG-VQA reward ablation without the Soft EG-F1 component."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eg_vqa_reward import compute_score as compute_full_score  # noqa: E402


def compute_score(
    predict_str: str,
    ground_truth: Any,
    format_weight: float = 0.1,
    evidence_weight: float = 0.3,
    accuracy_weight: float = 0.6,
    iou_threshold: float = 0.5,
    sim_threshold: float = 0.75,
    semantic_model_dir: str = "./Science_Bert",
    semantic_device: str = "cuda",
) -> Dict[str, float]:
    """Use hard EG-F1 while preserving format and answer rewards."""
    return compute_full_score(
        predict_str=predict_str,
        ground_truth=ground_truth,
        format_weight=format_weight,
        evidence_weight=evidence_weight,
        accuracy_weight=accuracy_weight,
        iou_threshold=iou_threshold,
        sim_threshold=sim_threshold,
        semantic_model_dir=semantic_model_dir,
        semantic_device=semantic_device,
        use_soft_matching=False,
    )

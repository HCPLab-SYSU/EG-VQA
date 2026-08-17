import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


# Dynamic score functions are loaded outside normal package context.
# Add the current directory to sys.path so sibling imports work.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from evidence_reward import evidence_reward, parse_ground_truth, parse_model_response  # noqa: E402


# Enforce the exact output order: evidence -> think -> answer.
OUTPUT_PATTERN = re.compile(
    r"^\s*<evidence>.*?</evidence>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)
ANSWER_JUDGE_PATTERN = re.compile(r"Answer:\s*(\d+(?:\.\d+)?)")


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")
_THREAD_LOCAL = threading.local()


def get_openai_client() -> OpenAI:
    """Use one client per thread to avoid shared-client contention."""
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for the answer-judge reward.")
        client_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        client = OpenAI(**client_kwargs)
        _THREAD_LOCAL.client = client
    return client


def create_evaluation_prompt(
    metadata: str,
    question: str,
    golden_answer: str,
    model_answer: str,
    title: str | None = None,
) -> str:
    """Build the judge prompt, with an optional video title."""
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
Answer: 0
Answer: 0.5
Answer: 1
"""
        return prompt.format(
            title=title,
            metadata=metadata,
            question=question,
            golden_answer=golden_answer,
            model_answer=model_answer,
        )

    prompt = """You are an expert specializing in evaluating whether a respondent's answer after watching a video matches the golden answer. We will provide the video segments' descriptions, question, golden answer, and the response to be judged below.

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
Answer: 0
Answer: 0.5
Answer: 1
"""
    return prompt.format(
        metadata=metadata,
        question=question,
        golden_answer=golden_answer,
        model_answer=model_answer,
    )


def normalize_generated_text(text: str) -> str:
    """Normalize spaces around XML-like tags such as <answer>."""
    text = "" if text is None else str(text)
    return re.sub(r"\s*(<|>|/)\s*", r"\1", text)


def extract_score_from_response(response_content: str) -> float | None:
    """Extract a numeric score from the judge model response."""
    try:
        for line in response_content.strip().splitlines():
            if line.startswith("Answer:"):
                score_str = line.split(":", 1)[1].strip().split()[0]
                return float(score_str)
        if "0.5" in response_content:
            return 0.5
        if "1" in response_content:
            return 1.0
        if "0" in response_content:
            return 0.0
        return None
    except Exception:
        return None


def generate_qwen(answer_text: str, parsed_ground_truth: Dict[str, Any]) -> str:
    """Call the external judge model and return the judge output."""
    golden_answer = parsed_ground_truth["answer"]

    if len(answer_text.split()) > 30:
        answer_text = " ".join(answer_text.split()[:30])

    question = parsed_ground_truth["question"]
    title = parsed_ground_truth.get("title")
    video_segments = parsed_ground_truth.get("video_segments", [])
    metadata = "\n".join(f"- {segment.strip()}" for segment in video_segments if str(segment).strip())

    prompt = create_evaluation_prompt(
        metadata=metadata,
        question=question,
        golden_answer=golden_answer,
        model_answer=answer_text,
        title=title,
    )
    messages = [{"role": "user", "content": prompt}]

    retries = 0
    while retries < 3:
        try:
            completion = get_openai_client().chat.completions.create(
                model="gemini-2.5-pro",
                temperature=0,
                messages=messages,
                timeout=30 
            )
            print(f"打分结果:\n{completion.choices[0].message.content}\n")
            return completion.choices[0].message.content
        except Exception as e:
            retries += 1
            print(f"judge model call failed: {e}. retry {retries}/3")
            time.sleep(3)

    return ""


def format_reward(predict_str: str) -> float:
    """Return 1 only when the full three-block output format is satisfied."""
    return 1.0 if OUTPUT_PATTERN.match(predict_str) else 0.0


def accuracy_reward(predict_str: str, ground_truth: Any) -> float:
    """Score the answer block with the external judge model."""
    parsed_ground_truth = parse_ground_truth(ground_truth)
    golden_answer = parsed_ground_truth["answer"]
    if not golden_answer:
        return 0.0

    # 如果修补后仍然提取不出 <answer> 内容，则答案奖励直接给 0。
    # 这样既不会报错，也不会把整段脏文本送去 judge。
    parsed_response, _ = parse_model_response(predict_str)
    if not parsed_response["answer"].strip():
        return 0.0

    answer_text = parsed_response["answer"].strip()
    judge_response = generate_qwen(answer_text, parsed_ground_truth)
    score = extract_score_from_response(judge_response)
    if score is None:
        return 0.0

    answer_length_limit = 5 * max(len(golden_answer), 1)
    if score in (1.0, 0.5) and len(answer_text) > answer_length_limit:
        return 0.1
    return score


def compute_base_score(
    predict_str: str,
    ground_truth: Any,
    format_weight: float = 0.1,
    evidence_weight: float = 0.3,
    accuracy_weight: float = 0.6,
    iou_threshold: float = 0.5,
    sim_threshold: float = 0.75,
    semantic_model_dir: str = "./Science_Bert",
    semantic_device: str = "cuda",
    use_soft_matching: bool = True,
) -> Dict[str, float]:
    """Compute the non-API parts of the reward serially."""
    del format_weight, evidence_weight, accuracy_weight
    predict_str = normalize_generated_text(predict_str)
    format_score = format_reward(predict_str)
    evidence_score = evidence_reward(
        predict_str,
        ground_truth,
        iou_threshold=iou_threshold,
        sim_threshold=sim_threshold,
        semantic_model_dir=semantic_model_dir,
        semantic_device=semantic_device,
        use_soft_matching=use_soft_matching,
    )
    return {
        "format": format_score,
        "evidence": evidence_score,
    }


def compute_accuracy_score(
    predict_str: str,
    ground_truth: Any,
    format_weight: float = 0.1,
    evidence_weight: float = 0.3,
    accuracy_weight: float = 0.6,
    iou_threshold: float = 0.5,
    sim_threshold: float = 0.75,
    semantic_model_dir: str = "./Science_Bert",
    semantic_device: str = "cuda",
) -> float:
    """Compute only the API-backed accuracy score."""
    del format_weight, evidence_weight, accuracy_weight
    del iou_threshold, sim_threshold, semantic_model_dir, semantic_device
    predict_str = normalize_generated_text(predict_str)
    return accuracy_reward(predict_str, ground_truth)


def combine_score(
    base_score: Dict[str, float],
    accuracy_score: float,
    format_weight: float = 0.1,
    evidence_weight: float = 0.3,
    accuracy_weight: float = 0.6,
    iou_threshold: float = 0.5,
    sim_threshold: float = 0.75,
    semantic_model_dir: str = "./Science_Bert",
    semantic_device: str = "cuda",
) -> Dict[str, float]:
    """Combine serial base scores with the concurrently computed accuracy score."""
    del iou_threshold, sim_threshold, semantic_model_dir, semantic_device
    total_weight = format_weight + evidence_weight + accuracy_weight
    if total_weight <= 0:
        raise ValueError("format_weight + evidence_weight + accuracy_weight must be positive.")

    format_score = base_score["format"]
    evidence_score = base_score["evidence"]
    overall = (
        format_weight * format_score
        + evidence_weight * evidence_score
        + accuracy_weight * accuracy_score
    ) / total_weight
    return {
        "overall": overall,
        "format": format_score,
        "evidence": evidence_score,
        "accuracy": accuracy_score,
    }


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
    use_soft_matching: bool = True,
) -> Dict[str, float]:
    """Main reward entry used by the training framework."""
    base_score = compute_base_score(
        predict_str,
        ground_truth,
        format_weight=format_weight,
        evidence_weight=evidence_weight,
        accuracy_weight=accuracy_weight,
        iou_threshold=iou_threshold,
        sim_threshold=sim_threshold,
        semantic_model_dir=semantic_model_dir,
        semantic_device=semantic_device,
        use_soft_matching=use_soft_matching,
    )
    accuracy_score = compute_accuracy_score(
        predict_str,
        ground_truth,
        format_weight=format_weight,
        evidence_weight=evidence_weight,
        accuracy_weight=accuracy_weight,
        iou_threshold=iou_threshold,
        sim_threshold=sim_threshold,
        semantic_model_dir=semantic_model_dir,
        semantic_device=semantic_device,
    )
    return combine_score(
        base_score,
        accuracy_score,
        format_weight=format_weight,
        evidence_weight=evidence_weight,
        accuracy_weight=accuracy_weight,
        iou_threshold=iou_threshold,
        sim_threshold=sim_threshold,
        semantic_model_dir=semantic_model_dir,
        semantic_device=semantic_device,
    )

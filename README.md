# EG-VQA: Benchmarking Verifiable Video Question Answering with Grounded Temporal Evidence

This repository contains the training and evaluation code for EG-VQA and EG-Reasoner.

Paper: [EG-VQA: Benchmarking Verifiable Video Question Answering with Grounded Temporal Evidence](https://arxiv.org/abs/2606.24797)

Dataset: [lphuang33/EG-VQA](https://huggingface.co/datasets/lphuang33/EG-VQA)

## Overview

EG-VQA is an evidence-grounded video question answering benchmark. Each question is paired with temporal evidence annotations, requiring a model to answer the question and identify the relevant video segments.

The benchmark contains:

- 2,067 videos;
- 11,838 question-answer pairs;
- 8,949 training questions from 1,731 videos;
- 2,889 test questions from 336 videos;
- temporal evidence annotations with `timestamp` and `description` fields.

EG-Reasoner is trained with outcome-based rewards for answer correctness, evidence formatting, temporal alignment, and semantic consistency.

## Repository Structure

```text
EG-VQA/
├── README.md
├── data/
│   ├── train.json
│   ├── test.json
│   ├── videos/                         # User-provided original videos
│   ├── frames/
│   │   ├── train/                      # Generated training frames
│   │   └── test/                       # Generated test frames
│   └── train_set_parquet/              # Generated training Parquet shards
├── preprocess/
│   ├── video_clip.py
│   └── prepare_train_parquet.py
├── EG-Reasoner/
│   ├── examples/
│   │   ├── train_eg_vqa.sh             # Main training entry point
│   │   ├── eg_vqa.yaml                 # Main training configuration
│   │   ├── ablations/                  # Ablation training scripts
│   │   └── score_function/             # Reward functions
│   ├── scripts/
│   │   └── model_merger.py             # Merge FSDP checkpoints
│   ├── Science_Bert/                   # BGE-M3 semantic model
│   ├── checkpoints/                    # Generated training checkpoints
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── setup.py
│   └── verl/                           # veRL training framework
└── evaluation/
    ├── evaluate.py
    ├── answer_metrics.py
    ├── evidence_metrics.py
    └── results/                        # Generated evaluation results
```

Directories marked as generated or user-provided are not included in the repository. `evaluation/evaluate.py` runs model inference. `answer_metrics.py` computes relaxed and strict answer accuracy, while `evidence_metrics.py` computes IoU-F1 and EG-F1.

## Data Acquisition

The original video files are not included. Before preprocessing, download the corresponding videos from their original source datasets and place them in the `data/videos/` folder, at the same level as `train.json` and `test.json`.

## Video Preprocessing

Install the dependencies from `EG-Reasoner/requirements.txt`, then run the preprocessing scripts from the project root:

```bash
python preprocess/video_clip.py \
  --dataset data/train.json \
  --video-root data \
  --output-root data/frames/train
python preprocess/video_clip.py \
  --dataset data/test.json \
  --video-root data \
  --output-root data/frames/test
python preprocess/prepare_train_parquet.py
```

The first command extracts up to 32 timestamped frames for each training video. The second command extracts test frames. The third command converts only the training split into Parquet shards.

The generated files are:

```text
data/frames/train/<video_id>/*.png
data/frames/test/<video_id>/*.png
data/train_set_parquet/part-*.parquet
```

No test Parquet is required by the current evaluation design.

## Requirements

```bash
conda create -n eg-reasoner python=3.10
conda activate eg-reasoner
cd EG-Reasoner
pip install -e .
```

The training framework is based on veRL. The internal Python package name `verl` is intentionally unchanged because it is used by the framework imports.

The training reward and evidence evaluation use BGE-M3 (`BAAI/bge-m3`) for semantic similarity. The default local directory is:

```text
EG-Reasoner/Science_Bert/
```

If the model is not found there, the code downloads it automatically on first use. This requires access to Hugging Face and sufficient disk space. For offline use, download BGE-M3 in advance and place the complete Sentence-Transformers model files directly in `EG-Reasoner/Science_Bert/`. The location can be changed with `SEMANTIC_MODEL_DIR` in `examples/train_eg_vqa.sh` or `--semantic-model-dir` during evaluation.

## Training

We provide the training code for EG-Reasoner. You can use the following command to run the training code.

```bash
cd EG-Reasoner
bash examples/train_eg_vqa.sh
```

## Merge FSDP Checkpoints

The training framework may save model parameters as FSDP DTensor shards. These shards need to be merged before the model can be used as a normal Hugging Face checkpoint.

The merge script expects an actor checkpoint directory containing both shard files and a Hugging Face configuration directory.

When trainer.save_checkpoint_path is null, veRL builds the checkpoint path relative to the EG-Reasoner directory:

    checkpoints/<project_name>/<experiment_name>/global_step_<step>/actor/

Here, project_name and experiment_name come from trainer.project_name and trainer.experiment_name. For the main experiment in this repository, they are EG-VQA and EG-Reasoner, respectively.

```text
actor/
├── model_world_size_4_rank_0.pt
├── model_world_size_4_rank_1.pt
├── ...
└── huggingface/
    └── config.json
```

Run:

```bash
cd EG-Reasoner
PROJECT_NAME=EG-VQA
EXPERIMENT_NAME=EG-Reasoner
STEP=20
python scripts/model_merger.py \
  --local_dir "checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${STEP}/actor"
```

The merged model is written back to:

```text
checkpoints/<project_name>/<experiment_name>/global_step_<step>/actor/huggingface/
```

The current script supports FSDP and DDP+FSDP checkpoints. FSDP+Tensor-Parallel checkpoints are not supported by this merger.

## Inference and Evaluation

By default, the evaluator searches for the latest merged actor model under:

    EG-Reasoner/checkpoints/<project_name>/<experiment_name>/global_step_<step>/actor/huggingface/

The evaluator always uses the latest checkpoint. A direct model directory remains supported through --model-path.

After training and merging a checkpoint, run:

```bash
cd EG-VQA
python evaluation/evaluate.py \
  --annotations data/test.json \
  --frames-root data/frames/test
```

The evaluator processes the complete test split. Each run is saved as one JSON file, and an interrupted run leaves a .partial.json checkpoint that can be resumed with the same command.

```text
evaluation/results/<project_name>_<experiment_name>_step_<step>.json
evaluation/results/<project_name>_<experiment_name>_step_<step>.partial.json
```

The evaluator uses one GPU by default. To use two GPUs for vLLM tensor parallelism, run:

    python evaluation/evaluate.py --annotations data/test.json --frames-root data/frames/test --cuda-visible-devices 0,1 --tensor-parallel-size 2

The intended evaluation inputs are:

```text
data/test.json
data/frames/test/
EG-Reasoner/checkpoints/<project_name>/<experiment_name>/global_step_<step>/actor/huggingface/
```

### Answer and evidence metrics

The inference JSON already contains both the public ground-truth fields and the model response. Therefore, no intermediate extraction script is needed:

```text
answer/evidence ground truth:  result["answer"], result["evidence"]
model answer/evidence:         result["parsed_response"]["answer"], result["parsed_response"]["evidence"]
```

Set an OpenAI-compatible API key before computing answer accuracy. `OPENAI_BASE_URL` is optional, and `OPENAI_MODEL` can be used to change the judge model.

```bash
python evaluation/answer_metrics.py \
  --input evaluation/results/<run_name>.json \
  --api-key "<your-api-key>" \
  --api-base "<your-openai-compatible-base-url>"
```

`--api-base` is optional for the official OpenAI endpoint. The equivalent environment variables `OPENAI_API_KEY` and `OPENAI_BASE_URL` are also supported. The script evaluates the complete test result, saves `<run_name>_answer_metrics.json`, and reports both relaxed accuracy (scores 0/0.5/1) and strict accuracy (only score 1 counts as correct). It can resume from `<run_name>_answer_metrics.partial.json` if an API call is interrupted.

Evidence metrics are computed in one run. The default settings are the four event-level IoU thresholds `0.1/0.3/0.5/0.7` and the three paper EG-F1 settings `(0.3, 0.5)`, `(0.3, 0.75)`, and `(0.5, 0.75)` for `(IoU threshold, semantic-similarity threshold)`:

```bash
python evaluation/evidence_metrics.py \
  --input evaluation/results/<run_name>.json \
  --semantic-device cuda
```

The BGE-M3 model is loaded from `EG-Reasoner/Science_Bert/` and downloaded there if it is not present. Results are saved as `<run_name>_evidence_metrics.json`. To evaluate a different threshold set without editing the script, use `--eg-thresholds` and `--iou-thresholds`.

## Citation

If you use this code for your research, please cite our paper.

```bibtex
@article{huang2026egvqa,
  title={EG-VQA: Benchmarking Verifiable Video Question Answering with Grounded Temporal Evidence},
  author={Huang, Linpeng and Chen, Weixing and Chen, Zexin and Liu, Yang and Lin, Liang},
  journal={arXiv preprint arXiv:2606.24797},
  year={2026}
}
```

If you have any questions about this code, please feel free to reach us at [huanglp33@mail2.sysu.edu.cn](mailto:huanglp33@mail2.sysu.edu.cn) or [liuy856@mail.sysu.edu.cn](mailto:liuy856@mail.sysu.edu.cn).

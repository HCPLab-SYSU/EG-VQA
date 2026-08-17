# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq


SHARD_FILENAME_PATTERN = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement) -> torch.Tensor:
    """Merge local tensors according to their DTensor placement."""
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial placement is not supported.")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise ValueError(f"Unsupported placement: {placement}")


def discover_shards(local_dir: Path) -> Tuple[int, List[Path]]:
    """Find one complete group of model shard files."""
    grouped_shards: Dict[int, Dict[int, Path]] = {}
    for path in local_dir.glob("model_world_size_*_rank_*.pt"):
        match = SHARD_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        world_size = int(match.group(1))
        rank = int(match.group(2))
        grouped_shards.setdefault(world_size, {})[rank] = path

    if not grouped_shards:
        raise FileNotFoundError(
            f"No model shards found in {local_dir}. Expected files named "
            "model_world_size_<N>_rank_<R>.pt."
        )
    if len(grouped_shards) != 1:
        raise ValueError(f"Found multiple model world sizes in {local_dir}: {sorted(grouped_shards)}")

    world_size, shards_by_rank = next(iter(grouped_shards.items()))
    ranks = sorted(shards_by_rank)
    expected_ranks = list(range(len(ranks)))
    if ranks != expected_ranks:
        raise ValueError(f"Shard ranks are not contiguous: found {ranks}")
    if len(ranks) != world_size:
        raise ValueError(f"Expected {world_size} shard files, found {len(ranks)}")

    return world_size, [shards_by_rank[rank] for rank in ranks]


def load_state_dict(path: Path) -> Dict[str, object]:
    """Load a checkpoint shard on CPU with DTensor metadata preserved."""
    return torch.load(path, map_location="cpu", weights_only=False)


def select_shard_paths(shard_paths: List[Path], mesh_dim_names: Tuple[str, ...], mesh) -> List[Path]:
    """Select the unique FSDP replica when the mesh also contains DDP."""
    if mesh_dim_names == ("fsdp",):
        expected_shards = mesh.shape[-1]
        if len(shard_paths) != expected_shards:
            raise ValueError(
                f"Checkpoint has {len(shard_paths)} files, but the FSDP mesh requires {expected_shards}."
            )
        return shard_paths

    if mesh_dim_names == ("ddp", "fsdp"):
        fsdp_shards = mesh.shape[-1]
        expected_files = mesh.shape[0] * mesh.shape[1]
        if len(shard_paths) != expected_files:
            raise ValueError(
                f"Checkpoint has {len(shard_paths)} files, but the DDP+FSDP mesh requires {expected_files}."
            )
        # veRL writes ranks in mesh order. The first DDP replica contains one
        # complete set of FSDP shards; the other replicas are duplicates.
        return shard_paths[:fsdp_shards]

    raise NotImplementedError(
        f"Unsupported device mesh {mesh_dim_names}. Only FSDP and DDP+FSDP are supported; "
        "FSDP+TP checkpoints are not supported."
    )


def get_model_class(config):
    """Select a compatible Hugging Face model class from the config."""
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        raise ValueError("The Hugging Face config does not define an architecture.")

    architecture = architectures[0]
    if "ForTokenClassification" in architecture:
        return AutoModelForTokenClassification
    if "ForCausalLM" in architecture:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in architecture:
        return AutoModelForVision2Seq
    raise NotImplementedError(f"Unknown architecture: {architectures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge veRL FSDP model shards into a Hugging Face checkpoint.")
    parser.add_argument(
        "--local_dir",
        required=True,
        type=Path,
        help="Checkpoint actor directory containing model_world_size_*_rank_*.pt and huggingface/.",
    )
    parser.add_argument(
        "--hf_upload_path",
        default=None,
        type=str,
        help="Optional Hugging Face model repository, for example username/eg-reasoner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir: Path = args.local_dir.resolve()

    if local_dir.name == "huggingface":
        raise ValueError("--local_dir must be the actor checkpoint directory, not its huggingface/ subdirectory.")
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {local_dir}")

    hf_path = local_dir / "huggingface"
    if not (hf_path / "config.json").is_file():
        raise FileNotFoundError(f"Expected Hugging Face config at {hf_path / 'config.json'}")

    world_size, all_shard_paths = discover_shards(local_dir)
    rank0_state_dict = load_state_dict(all_shard_paths[0])

    dtensor_keys = [key for key, value in rank0_state_dict.items() if isinstance(value, DTensor)]
    if not dtensor_keys:
        raise ValueError("Rank 0 contains no DTensor parameters; this script expects an FSDP checkpoint.")

    pivot_key = sorted(dtensor_keys)[0]
    pivot_weight = rank0_state_dict[pivot_key]
    device_mesh = pivot_weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = tuple(device_mesh.mesh_dim_names or ())
    print(f"Found world size {world_size}, device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    shard_paths = select_shard_paths(all_shard_paths, mesh_dim_names, mesh)
    state_dicts: List[Dict[str, object]] = [rank0_state_dict]

    with ThreadPoolExecutor(max_workers=max(1, min(32, os.cpu_count() or 1))) as executor:
        futures = [executor.submit(load_state_dict, path) for path in shard_paths[1:]]
        for future in futures:
            state_dicts.append(future.result())

    reference_keys = set(state_dicts[0])
    for rank, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict) != reference_keys:
            raise ValueError(f"Shard {rank} has a different set of parameter keys.")

    merged_state_dict: Dict[str, torch.Tensor] = {}
    parameter_placements: Dict[str, Tuple[Placement, ...]] = {}

    for key in sorted(reference_keys):
        tensors = [state_dict[key] for state_dict in state_dicts]
        dtensor_flags = [isinstance(tensor, DTensor) for tensor in tensors]

        if not any(dtensor_flags):
            merged_state_dict[key] = tensors[0]
            continue
        if not all(dtensor_flags):
            raise ValueError(f"Parameter {key} is DTensor in only some checkpoint shards.")

        local_tensors = []
        placements = None
        for tensor in tensors:
            assert isinstance(tensor, DTensor)
            current_placements = tuple(tensor.placements)
            if mesh_dim_names and mesh_dim_names[0] == "ddp":
                current_placements = current_placements[1:]
            if placements is None:
                placements = current_placements
            elif placements != current_placements:
                raise ValueError(f"Inconsistent placements for parameter {key}.")
            local_tensors.append(tensor.to_local())

        assert placements is not None
        parameter_placements[key] = placements
        if len(placements) != 1:
            raise NotImplementedError(f"Parameter {key} has unsupported placements: {placements}")
        merged_state_dict[key] = merge_by_placement(local_tensors, placements[0])

    print("Writing merged Hugging Face checkpoint")
    config = AutoConfig.from_pretrained(str(hf_path))
    auto_model = get_model_class(config)

    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.save_pretrained(str(hf_path), state_dict=merged_state_dict)
    print(f"Saved merged model to {hf_path}")

    if args.hf_upload_path:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
        api.upload_folder(folder_path=str(hf_path), repo_id=args.hf_upload_path, repo_type="model")
        print(f"Uploaded merged model to {args.hf_upload_path}")


if __name__ == "__main__":
    main()

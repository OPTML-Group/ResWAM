"""Distributed torchrun FSDP trainer for ResWAM across LIBERO, RoboTwin 2.0, and RoboDojo.

Supports:
  1. Online mode (default): computes frozen DINOv3 features on-the-fly for ``o_t`` and ``o_{t+H}``
     during training without requiring pre-extracted caches.
  2. Cached mode (``--use-dino-cache`` or ``--dino-cache-root``): loads pre-extracted DINOv3
     features from memory-mapped FP32 shards for maximum throughput.
  3. Optional pi0.5 shared temporal image augmentation (``--pi05-aug``).

Example launch:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 -m reswam.engine.trainer \
        --config configs/robotwin2.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ..data.codec import ZScoreCodec
from ..data.dataset import LeRobotWindowDataset, build_loader, isolated_loader_iterator
from ..models.dit_modules import assert_zero_init
from ..models.system import ResWAMSystem, system_config
from ..utils.config import load_config
from ..utils.dist import configure_determinism, init_distributed, log_json, virtual_rank_seed
from . import checkpoint as ckpt
from .fsdp import (
    assert_optimizer_fp32_contract,
    assert_recomputation_disabled,
    assert_trainable_parameter_dtype,
    build_optimizer,
    prepare_fp32_trainable_parameters,
    wrap_fsdp,
)
from .schedule import build_scheduler


def resolve_run_settings(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve training, data, and contract configurations from YAML or run_config.json."""
    cfg_path = args.config or args.run_config
    if cfg_path is None:
        raise ValueError("Provide either --config <yaml> or --run-config <json>")
    raw_cfg = load_config(cfg_path)

    if "dataset_config" in raw_cfg and "training" not in raw_cfg:
        run_cfg = dict(raw_cfg)
        data_cfg = load_config(run_cfg["dataset_config"])
        contract = load_config(data_cfg["contract"])
    else:
        data_section = raw_cfg.get("data", {})
        train_section = raw_cfg.get("training", {})
        model_section = raw_cfg.get("model", {})
        flow_section = model_section.get("flow", {})
        prepared_dir = Path(args.prepared_dir or data_section.get("prepared_dir", "./data/prepared"))
        reswam_json = (
            Path(args.dataset_config).resolve()
            if args.dataset_config
            else (prepared_dir / "reswam.json")
        )
        if reswam_json.is_file():
            data_cfg = load_config(reswam_json)
            contract = load_config(data_cfg["contract"])
        else:
            contract_file = prepared_dir / "dataset.json"
            stats_file = prepared_dir / "codec_statistics.json"
            if not contract_file.is_file() or not stats_file.is_file():
                raise FileNotFoundError(
                    f"Prepared dataset metadata not found in {prepared_dir}. "
                    f"Run the corresponding scripts/data_prep/prepare_*.py script first."
                )
            contract = load_config(contract_file)
            clean_file = prepared_dir / "clean_index.json"
            data_cfg = {
                "dataset_id": raw_cfg.get("dataset_id", contract.get("dataset_id", "dataset")),
                "contract": str(contract_file),
                "codec_statistics": str(stats_file),
                "action_dim": int(data_section.get("action_dim", contract["action_dim"])),
                "state_dim": int(data_section.get("state_dim", contract["state_dim"])),
                "horizon": int(data_section.get("horizon", 32)),
                "views": list(data_section.get("views", ["head", "left_wrist"])),
                "world_views": list(data_section.get("world_views", ["head"])),
                "instructions": contract.get("tasks", {}),
                "end_padding": data_section.get("end_padding", "repeat_last_frame_supervised"),
                **({"clean_index": str(clean_file)} if clean_file.is_file() else {}),
            }

        pi05_aug = bool(args.pi05_aug or data_section.get("pi05_aug", False))
        world_loss_weight = (
            float(args.world_loss_weight)
            if args.world_loss_weight is not None
            else float(flow_section.get("world_loss_weight", 0.1))
        )
        state_dropout_prob = (
            float(args.state_dropout_prob)
            if args.state_dropout_prob is not None
            else float(data_section.get("state_dropout_prob", data_cfg.get("state_dropout_prob", 0.0)))
        )
        task_index = (
            int(args.task_index)
            if args.task_index is not None
            else data_section.get("task_index", data_cfg.get("task_index"))
        )
        run_cfg = {
            "run_name": args.run_name or raw_cfg.get("dataset_id", "reswam"),
            "dataset_id": data_cfg.get("dataset_id", raw_cfg.get("dataset_id", "dataset")),
            "output_dir": str(args.output_dir or train_section.get("output_dir", "./outputs/reswam")),
            "seed": int(args.seed or train_section.get("seed", 20260907)),
            "per_gpu_batch": int(args.per_gpu_batch or train_section.get("per_gpu_batch", 32)),
            "grad_accum": int(args.grad_accum or train_section.get("grad_accum", 2)),
            "steps": int(args.steps or train_section.get("steps", 100000)),
            "warmup_steps": int(args.warmup_steps or train_section.get("warmup_steps", 1500)),
            "lr_schedule": str(args.lr_schedule or train_section.get("lr_schedule", "warmup_cosine")),
            "start_step": int(
                args.start_step if args.start_step is not None else train_section.get("start_step", 0)
            ),
            "init_from": (
                str(args.init_from)
                if args.init_from
                else (str(train_section["init_from"]) if train_section.get("init_from") else None)
            ),
            "save_every": int(args.save_every or train_section.get("save_every", 2000)),
            "save_full": not args.no_save_full and bool(train_section.get("save_full", True)),
            "keep_resume_states": int(train_section.get("keep_resume_states", 5)),
            "num_workers": int(
                args.num_workers
                if args.num_workers is not None
                else train_section.get("num_workers", 8)
            ),
            "stop_after": args.stop_after,
            "world_loss_weight": world_loss_weight,
            "action_tau_shift": float(flow_section.get("action_tau_shift", 1.0)),
            "world_tau_shift": float(flow_section.get("world_tau_shift", 5.0)),
            "solver_steps": int(flow_section.get("solver_steps", 10)),
            "state_dropout_prob": state_dropout_prob,
            "pi05_aug": pi05_aug,
            **({"task_index": task_index} if task_index is not None else {}),
            "prompt_style": model_section.get("rynnbrain", {}).get("prompt_style", "multi_camera"),
            "expert_layout": model_section.get("expert", {}).get("layout"),
            "use_state_null_tokens": model_section.get("expert", {}).get("use_state_null_tokens"),
            "rynnbrain_path": model_section.get("rynnbrain", {}).get(
                "model_id", "Alibaba-DAMO-Academy/RynnBrain1.1-2B"
            ),
            "dino_path": model_section.get("dino", {}).get(
                "model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m"
            ),
            "optimizer": train_section.get("optimizer", {}),
            "wandb": {
                "enabled": bool(args.wandb or raw_cfg.get("wandb", {}).get("enabled", False)),
                "project": raw_cfg.get("wandb", {}).get(
                    "project", f"reswam_{raw_cfg.get('dataset_id', 'train')}"
                ),
                "id": raw_cfg.get("wandb", {}).get("id"),
            },
        }
        use_cache = bool(args.use_dino_cache or data_section.get("use_dino_cache", False))
        if pi05_aug:
            use_cache = False
        cache_root = (
            args.dino_cache_root
            or data_section.get("dino_cache_root")
            or os.environ.get("RESWAM_DINO_CACHE_ROOT")
        )
        if use_cache and cache_root:
            run_cfg["dino_cache_root"] = str(cache_root)
        elif args.dino_cache_root and not pi05_aug:
            run_cfg["dino_cache_root"] = str(args.dino_cache_root)

    if args.dataset_root:
        contract = {**contract, "root": str(Path(args.dataset_root).resolve())}
    elif os.environ.get("RESWAM_LOCAL_DATASET_ROOT"):
        local_root = Path(os.environ["RESWAM_LOCAL_DATASET_ROOT"])
        if local_root.is_dir():
            contract = {**contract, "root": str(local_root.resolve())}

    return run_cfg, data_cfg, contract


def main() -> None:
    ap = argparse.ArgumentParser(description="ResWAM Distributed FSDP Trainer")
    ap.add_argument("--config", type=Path, help="Path to benchmark YAML config (e.g. configs/robotwin2.yaml)")
    ap.add_argument("--run-config", type=Path, help="Path to resolved JSON run config")
    ap.add_argument("--prepared-dir", type=Path, help="Override prepared dataset directory")
    ap.add_argument("--dataset-config", type=Path, help="Override prepared reswam.json path")
    ap.add_argument("--dataset-root", type=Path, help="Override raw LeRobot dataset root")
    ap.add_argument("--dino-cache-root", type=Path, help="Optional pre-extracted DINOv3 cache root")
    ap.add_argument("--use-dino-cache", action="store_true", help="Enable loading precomputed DINOv3 cache")
    ap.add_argument("--pi05-aug", action="store_true", help="Enable pi0.5 shared temporal image augmentation")
    ap.add_argument("--world-loss-weight", type=float, default=None, help="Override world residual loss weight lambda_w")
    ap.add_argument("--state-dropout-prob", type=float, default=None, help="Override proprioceptive state dropout probability")
    ap.add_argument("--task-index", type=int, default=None, help="Filter training dataset to a single task_index")
    ap.add_argument("--output-dir", type=Path, help="Override training output directory")
    ap.add_argument("--run-name", type=str, help="Override run name")
    ap.add_argument("--steps", type=int, help="Override total optimizer steps")
    ap.add_argument("--warmup-steps", type=int, help="Override warmup steps")
    ap.add_argument(
        "--lr-schedule",
        choices=("warmup_cosine", "constant_floor"),
        help="LR schedule (constant_floor = constant min_lr continuation stage)",
    )
    ap.add_argument("--start-step", type=int, help="First optimizer update of a continuation stage")
    ap.add_argument("--init-from", type=Path, help="Stage-1 output dir to continue from (constant_floor)")
    ap.add_argument("--per-gpu-batch", type=int, help="Override per-GPU microbatch size")
    ap.add_argument("--grad-accum", type=int, help="Override gradient accumulation steps")
    ap.add_argument("--save-every", type=int, help="Override checkpoint interval")
    ap.add_argument("--stop-after", type=int, help="Stop after N optimizer steps")
    ap.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count")
    ap.add_argument("--seed", type=int, help="Override training seed")
    ap.add_argument("--no-save-full", action="store_true", help="Skip exporting full evaluation weights")
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = ap.parse_args()

    run_cfg, data_cfg, contract = resolve_run_settings(args)
    output = Path(run_cfg["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    world_size, rank, local_rank = init_distributed()
    ckpt.init_sync_group()
    seed = int(run_cfg["seed"])
    configure_determinism(seed, rank=rank)

    microbatch = int(run_cfg["per_gpu_batch"])
    accumulation = int(run_cfg["grad_accum"])
    steps = int(run_cfg["steps"])

    codec = ZScoreCodec.from_file(data_cfg["codec_statistics"])
    world_views = run_cfg.get("world_views") or data_cfg.get("world_views")
    task_index = (
        run_cfg.get("task_index")
        if run_cfg.get("task_index") is not None
        else data_cfg.get("task_index")
    )
    episode_indices = (
        run_cfg.get("episode_indices")
        if run_cfg.get("episode_indices") is not None
        else data_cfg.get("episode_indices")
    )
    clean_index_path = run_cfg.get("clean_index") or data_cfg.get("clean_index")
    clean_index_sha256 = run_cfg.get("clean_index_sha256") or data_cfg.get("clean_index_sha256")
    pi05_aug = bool(run_cfg.get("pi05_aug", data_cfg.get("pi05_aug", False)))
    dino_cache_root = (
        None
        if pi05_aug
        else (run_cfg.get("dino_cache_root") or data_cfg.get("dino_cache_root"))
    )
    dino_cache_manifests = run_cfg.get("dino_cache_manifests") or data_cfg.get(
        "dino_cache_manifests"
    )

    dataset = LeRobotWindowDataset(
        contract,
        codec,
        horizon=int(data_cfg["horizon"]),
        instructions={int(k): str(v) for k, v in data_cfg["instructions"].items()},
        world_views=world_views,
        task_index=task_index,
        episode_indices=episode_indices,
        dino_cache_root=dino_cache_root,
        dino_cache_manifests=dino_cache_manifests,
        clean_index_path=clean_index_path,
        clean_index_sha256=clean_index_sha256,
        end_padding=str(data_cfg.get("end_padding", "repeat_last_frame_supervised")),
        pi05_aug=pi05_aug,
        base_seed=seed,
    )
    loader, sampler = build_loader(
        dataset,
        rank=rank,
        world_size=world_size,
        seed=seed,
        microbatch=microbatch,
        num_workers=int(run_cfg["num_workers"]),
    )
    log_json(
        rank,
        {
            "event": "data",
            "episodes": len(dataset.episodes),
            "windows": len(dataset),
            "per_rank_per_epoch": len(sampler),
            "views": dataset.views,
            "world_views": dataset.world_views,
            "task_index": task_index,
            "clean_indexed": clean_index_path is not None,
            "pi05_aug": dataset.pi05_aug,
            "data_root": str(dataset.root),
            "dino_cached": dataset.dino_caches is not None,
            "updates_per_epoch": len(sampler) // (microbatch * accumulation),
        },
    )

    t0 = time.perf_counter()
    sys_cfg = system_config(
        action_dim=int(data_cfg["action_dim"]),
        state_dim=int(data_cfg["state_dim"]),
        horizon=int(data_cfg["horizon"]),
        views=data_cfg["views"],
        world_views=dataset.world_views,
        world_loss_weight=float(run_cfg.get("world_loss_weight", 0.1)),
        action_tau_shift=float(run_cfg.get("action_tau_shift", 1.0)),
        world_tau_shift=float(run_cfg.get("world_tau_shift", 5.0)),
        solver_steps=int(run_cfg.get("solver_steps", 10)),
        state_dropout_prob=float(run_cfg.get("state_dropout_prob", 0.0)),
        prompt_style=str(run_cfg.get("prompt_style", "multi_camera")),
        expert_layout=run_cfg.get("expert_layout"),
        use_state_null_tokens=run_cfg.get("use_state_null_tokens"),
        rynnbrain_path=str(
            run_cfg.get("rynnbrain_path", "Alibaba-DAMO-Academy/RynnBrain1.1-2B")
        ),
        dino_path=str(
            run_cfg.get("dino_path", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        ),
    )
    system = ResWAMSystem(sys_cfg)
    before_fsdp = prepare_fp32_trainable_parameters(system)
    assert_zero_init(system.objective.expert)
    system = system.to(local_rank)
    log_json(
        rank,
        {
            "event": "system_built",
            "seconds": round(time.perf_counter() - t0, 1),
            "parameters": sum(p.numel() for p in system.parameters()),
            "trainable": sum(p.numel() for p in system.parameters() if p.requires_grad),
            "before_fsdp": before_fsdp,
        },
    )

    wrapped = wrap_fsdp(system, local_rank, activation_checkpointing=False)
    assert_trainable_parameter_dtype(wrapped, stage="after_fsdp")
    checkpoint_targets = assert_recomputation_disabled(wrapped)
    vlm_ids = {id(p) for p in system.vlm.parameters()}
    vlm_params, expert_params = [], []
    for parameter in system.parameters():
        if parameter.requires_grad:
            (vlm_params if id(parameter) in vlm_ids else expert_params).append(parameter)

    opt_cfg = run_cfg.get("optimizer", {})
    optimizer = build_optimizer(
        vlm_params,
        expert_params,
        vlm_peak_lr=float(opt_cfg.get("vlm_peak_lr", 1e-5)),
        vlm_min_lr=float(opt_cfg.get("vlm_min_lr", 1e-6)),
        expert_peak_lr=float(opt_cfg.get("expert_peak_lr", 1e-4)),
        expert_min_lr=float(opt_cfg.get("expert_min_lr", 1e-5)),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-8)),
    )
    assert_optimizer_fp32_contract(optimizer, require_initialized=False)
    start_step = int(run_cfg.get("start_step", 0))
    scheduler = build_scheduler(
        optimizer,
        kind=str(run_cfg.get("lr_schedule", "warmup_cosine")),
        warmup=int(run_cfg["warmup_steps"]),
        total=steps,
        start=start_step,
    )
    torch.cuda.reset_peak_memory_stats()
    log_json(
        rank,
        {
            "event": "fsdp",
            "recompute_targets": checkpoint_targets,
            "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        },
    )

    step = consumed = 0
    resume_root = output
    if ckpt.latest_step(output) is None and run_cfg.get("init_from"):
        # Stage-2 continuation: start from the final resume state (model, optimizer, RNG,
        # data position) of the stage-1 run directory, with a migrated LR scheduler.
        resume_root = Path(run_cfg["init_from"]).resolve()
        if ckpt.latest_step(resume_root) != start_step:
            raise RuntimeError(
                f"init_from {resume_root} has latest step {ckpt.latest_step(resume_root)}, "
                f"expected start_step {start_step}"
            )
    if ckpt.latest_step(resume_root) is not None:
        step, position = ckpt.load(
            wrapped,
            optimizer,
            scheduler,
            output=resume_root,
            codec=codec.metadata,
            rank=rank,
            world_size=world_size,
        )
        sampler.load_state_dict(position)
        consumed = int(position["position"])
        log_json(
            rank,
            {
                "event": "resumed",
                "step": step,
                "data_position": position,
                "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
            },
        )

    wandb_run = None
    if rank == 0 and run_cfg.get("wandb", {}).get("enabled"):
        import wandb

        wandb_run = wandb.init(
            project=run_cfg["wandb"]["project"],
            name=run_cfg["run_name"],
            id=run_cfg["wandb"].get("id"),
            resume="allow",
            config=run_cfg,
            dir=str(output),
        )

    iterator = isolated_loader_iterator(loader, seed=seed, rank=rank, epoch=sampler.epoch)
    optimizer.zero_grad(set_to_none=True)
    stop_at = min(steps, int(run_cfg.get("stop_after") or steps))
    metrics_path = output / "metrics.jsonl"
    last = time.perf_counter()

    while step < stop_at:
        metrics: dict[str, torch.Tensor] = {}
        phase_data = phase_forward = phase_backward = 0.0
        for microstep in range(accumulation):
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                sampler.set_epoch(sampler.epoch + 1)
                consumed = 0
                iterator = isolated_loader_iterator(
                    loader, seed=seed, rank=rank, epoch=sampler.epoch
                )
                batch = next(iterator)
            phase_data += time.perf_counter() - start
            consumed += int(batch["actions"].shape[0])
            if int(batch["actions"].shape[0]) != microbatch:
                raise RuntimeError("runtime batch mismatch")
            context = wrapped.no_sync() if microstep < accumulation - 1 else nullcontext()
            with context:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = wrapped(
                    batch, seed=virtual_rank_seed(seed, rank, microstep, step, world_size)
                )
                torch.cuda.synchronize()
                phase_forward += time.perf_counter() - start
                if not torch.isfinite(result.loss):
                    raise FloatingPointError("non-finite loss")
                start = time.perf_counter()
                (result.loss / accumulation).backward()
                torch.cuda.synchronize()
                phase_backward += time.perf_counter() - start
            for name, value in result.metrics.items():
                metrics[name] = metrics.get(name, torch.zeros_like(value)) + value / accumulation

        start = time.perf_counter()
        grad_norm = wrapped.clip_grad_norm_(1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient")
        lrs = scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        phase_optimizer = time.perf_counter() - start
        step += 1

        names = list(metrics)
        values = torch.stack([metrics[n].float().mean() for n in names])
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(values)
            values /= world_size
        now = time.perf_counter()
        row = dict(zip(names, values.cpu().tolist()))
        row.update(lrs)
        row.update(
            step=step,
            epoch=sampler.epoch,
            data_position=consumed,
            gradient_norm=float(grad_norm),
            update_seconds=now - last,
            data_seconds=phase_data,
            forward_seconds=phase_forward,
            backward_seconds=phase_backward,
            optimizer_seconds=phase_optimizer,
            samples_per_second=microbatch * accumulation * world_size / (now - last),
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        peaks = torch.tensor(
            [row["cuda_peak_allocated_gib"], row["cuda_peak_reserved_gib"]], device=local_rank
        )
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        row["cuda_peak_allocated_gib_max_rank"], row["cuda_peak_reserved_gib_max_rank"] = (
            peaks.tolist()
        )
        last = now
        if rank == 0:
            print(json.dumps({"event": "update", **row}), flush=True)
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if wandb_run:
                wandb_run.log(row, step=step)

        if (step - start_step) % int(run_cfg["save_every"]) == 0 or (
            step == stop_at and (step < steps or run_cfg.get("save_final", True))
        ):
            ckpt.save(
                wrapped,
                optimizer,
                scheduler,
                output=output,
                step=step,
                data_position={"epoch": sampler.epoch, "position": consumed},
                codec=codec.metadata,
                config={"run": run_cfg, "system": sys_cfg, "dataset": data_cfg},
                rank=rank,
                world_size=world_size,
                full=bool(run_cfg["save_full"]),
                keep_resume_states=int(run_cfg["keep_resume_states"]),
            )

    if wandb_run:
        wandb_run.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end smoke and unit tests for the standalone ResWAM repository.

Verifies:
  1. YAML configuration loading and environment variable expansion for all 3 benchmarks
     (LIBERO-Plus, RoboTwin 2.0, and RoboDojo)
  2. ZScoreCodec, LiberoZScoreCodec, binary +-1 gripper mode, and 8-D -> 11-D 6D-rotation state featurization
  3. LiLa stationary-frame trajectory cleaning, outlier filtering, and B-spline action smoothing
  4. pi0.5 shared temporal image augmentation bit-exactness across (o_t, o_{t+H})
  5. ResWAMExpert construction, zero-initialization contract, forward pass, and shape checks
  6. ResWAMFlowObjective LayerNorm residual target, flow matching loss, and 10-step synchronous Euler sampling
  7. ResWAMSystem end-to-end forward pass (online DINOv3 mode and precomputed cache mode) and sample_actions
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import reswam  # noqa: E402
from reswam.data import (  # noqa: E402
    LiberoZScoreCodec,
    StateNormalizer11D,
    ZScoreCodec,
    augment_primary_pair_uint8,
    augment_window_cameras,
    augment_wrist_pair_uint8,
    build_libero_state_statistics,
    clean_episode,
    describe,
    featurize_libero_state_11d,
    float_view,
    window_positions,
)
from reswam.engine import (  # noqa: E402
    ConstantFloor,
    WarmupCosine,
    build_optimizer,
    prepare_fp32_trainable_parameters,
)
from reswam.models import (  # noqa: E402
    DINO_DIM,
    TOKENS_PER_VIEW,
    ResWAMExpert,
    ResWAMFlowObjective,
    ResWAMSystem,
    assert_zero_init,
    preprocess_camera,
    synchronous_sample,
    system_config,
)
from reswam.utils import bspline_smooth, load_config  # noqa: E402


class DummyVLM(nn.Module):
    """Lightweight mock VLM encoder for fast CPU/GPU unit testing without downloading 4 GB weights."""

    def __init__(self, hidden_size: int = 2048, seq_len: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.proj = nn.Linear(3, hidden_size)

    def prepare(
        self, cameras: Mapping[str, Sequence[np.ndarray]], instructions: Sequence[str]
    ) -> dict[str, torch.Tensor]:
        first_view = next(iter(cameras.values()))
        means = []
        for img in first_view:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            means.append(arr.mean(axis=(0, 1)))
        x = torch.tensor(np.stack(means), dtype=torch.float32)
        return {"mean_rgb": x, "batch_size": torch.tensor(len(instructions))}

    def forward(self, prepared: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.proj.weight.device
        x = prepared["mean_rgb"].to(device)
        batch = x.shape[0]
        tokens = self.proj(x)[:, None, :].expand(batch, self.seq_len, self.hidden_size).contiguous()
        mask = torch.ones((batch, self.seq_len), dtype=torch.bool, device=device)
        return tokens, mask


class DummyDINO(nn.Module):
    """Lightweight mock DINOv3 encoder mapping [B, 3, H, W] -> [B, 64, 768]."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.ones(1), requires_grad=False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(images.float(), (8, 8))
        flat = pooled.flatten(2).transpose(1, 2)  # [B, 64, 3]
        return flat.repeat(1, 1, DINO_DIM // 3).contiguous()


def test_configs_load_and_expand() -> None:
    expected = {
        "libero_plus.yaml": (256, 100000, 8, 7, 11, 0.3, 0.5),
        "robotwin2.yaml": (256, 100000, 32, 14, 14, 0.0, 0.1),
        "robodojo.yaml": (256, 200000, 32, 14, 14, 0.0, 0.1),
        "robodojo_stage2.yaml": (256, 400000, 32, 14, 14, 0.0, 0.1),
    }
    for name, (gbs, steps, horizon, act_dim, st_dim, st_drop, lw) in expected.items():
        cfg = load_config(REPO_ROOT / "configs" / name)
        assert "benchmark" in cfg
        assert "data" in cfg and "model" in cfg and "training" in cfg and "evaluation" in cfg
        assert int(cfg["training"]["global_batch_size"]) == gbs
        assert int(cfg["training"]["steps"]) == steps
        assert int(cfg["data"]["horizon"]) == horizon
        assert int(cfg["data"]["action_dim"]) == act_dim
        assert int(cfg["data"]["state_dim"]) == st_dim
        assert abs(float(cfg["data"]["state_dropout_prob"]) - st_drop) < 1e-9
        assert abs(float(cfg["model"]["flow"]["world_loss_weight"]) - lw) < 1e-9


def test_codecs_and_state_featurization() -> None:
    rng = np.random.default_rng(42)
    raw_libero_state = rng.standard_normal((128, 8)).astype(np.float32)
    feat_11d = featurize_libero_state_11d(raw_libero_state)
    assert feat_11d.shape == (128, 11)
    state_stats = build_libero_state_statistics(raw_libero_state)
    normalizer = StateNormalizer11D.from_dict(state_stats)
    norm_11d = normalizer(raw_libero_state)
    assert norm_11d.shape == (128, 11) and norm_11d.dtype == np.float32

    libero_codec = LiberoZScoreCodec(
        mean=tuple([0.0] * 6),
        std=tuple([1.0] * 6),
        state_normalizer=normalizer,
    )
    raw_act = np.zeros((8, 7), dtype=np.float32)
    raw_act[:, 6] = np.array([0.0, 1.0, 0.2, 0.8, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    enc_act = libero_codec.encode_action(raw_act)
    dec_act = libero_codec.decode_action(enc_act)
    assert np.allclose(dec_act[:, :6], raw_act[:, :6], atol=1e-6)
    assert set(np.unique(dec_act[:, 6]).tolist()) == {-1.0, 1.0}

    # Test unified ZScoreCodec in LIBERO binary_pm1 + 11D state mode
    libero_stats_payload = {
        "schema_version": "reswam-zscore-statistics-v1",
        "gripper_indices": [6],
        "state_gripper_indices": [],
        "gripper_mode": "binary_pm1",
        "encode_clamp_sigma": 12.0,
        "state_clamp_sigma": 10.0,
        "state_featurization": "xyz_rot6d_fingers2_11d",
        "action": describe(raw_act, [6], fixed_gripper_range=(0.0, 1.0)),
        "state": describe(feat_11d, []),
    }
    z_libero = ZScoreCodec.from_statistics(libero_stats_payload)
    assert z_libero.effective_state_clamp_sigma == 10.0
    assert z_libero.gripper_mode == "binary_pm1"
    enc_z = z_libero.encode_action(raw_act)
    dec_z = z_libero.decode_action(enc_z)
    assert np.allclose(dec_z[:, 6], np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))
    enc_st = z_libero.encode_state(raw_libero_state[:4])
    assert enc_st.shape == (4, 11) and np.allclose(enc_st, norm_11d[:4], atol=1e-6)

    # Test 14-D bimanual ZScoreCodec roundtrip
    raw_14d = rng.uniform(-1.5, 1.5, size=(200, 14)).astype(np.float32)
    raw_14d[:, [6, 13]] = rng.uniform(0.0, 1.0, size=(200, 2)).astype(np.float32)
    stats_payload = {
        "schema_version": "reswam-zscore-statistics-v1",
        "gripper_indices": [6, 13],
        "state_gripper_indices": [6, 13],
        "encode_clamp_sigma": 12.0,
        "action": describe(raw_14d, [6, 13], fixed_gripper_range=(0.0, 1.0)),
        "state": describe(raw_14d, [6, 13], fixed_gripper_range=(0.0, 1.0)),
    }
    codec_14d = ZScoreCodec.from_statistics(stats_payload)
    roundtrip = codec_14d.decode_action(codec_14d.encode_action(raw_14d))
    assert np.allclose(roundtrip, raw_14d, atol=1e-5)


def test_cleaning_bspline_and_pi05_augmentation() -> None:
    traj = np.linspace(0.0, 1.0, 50 * 14, dtype=np.float32).reshape(50, 14)
    traj[10:20] = traj[10]  # stationary segment
    res = clean_episode(traj)
    assert not res.outlier_episode
    assert len(res.keep_indices) < 50

    outlier_traj = traj.copy()
    outlier_traj[5, 0] = 3.5  # > pi on non-gripper joint
    out_res = clean_episode(outlier_traj)
    assert out_res.outlier_episode and len(out_res.keep_indices) == 0

    smoothed = bspline_smooth(traj[:32], degree=3, num_ctrl_pts=8)
    assert smoothed.shape == (32, 14) and np.isfinite(smoothed).all()

    pos = window_positions(length=40, start=35, horizon=8)
    assert len(pos["actions"]) == 8
    assert pos["endpoint"] == 39
    assert bool(pos["endpoint_padded"]) is True

    # Verify pi0.5 shared temporal augmentation: if o_t == o_{t+H}, aug(o_t) == aug(o_{t+H}) bit-for-bit
    frame = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8, generator=torch.Generator().manual_seed(99))
    pair = torch.stack([frame, frame.clone()], dim=0)
    aug_pair, _ = augment_primary_pair_uint8(pair, torch.Generator().manual_seed(1234))
    assert aug_pair.shape == (2, 3, 128, 128) and aug_pair.dtype == torch.uint8
    assert torch.equal(aug_pair[0], aug_pair[1])

    wrist_pair, _ = augment_wrist_pair_uint8(pair, torch.Generator().manual_seed(5678))
    assert wrist_pair.shape == (2, 3, 128, 128) and torch.equal(wrist_pair[0], wrist_pair[1])

    aug_cur, aug_end = augment_window_cameras(
        {"head": frame, "left_wrist": frame},
        {"head": frame.clone()},
        base_seed=20260830,
        epoch=3,
        current_row=10,
        endpoint_row=10,
    )
    assert torch.equal(aug_cur["head"], aug_end["head"])


def test_expert_objective_and_system_online_and_cached() -> None:
    torch.manual_seed(0)
    expert = ResWAMExpert(
        action_dim=14,
        state_dim=14,
        horizon=32,
        num_world_views=1,
        dim=256,
        heads=8,
        ffn_dim=512,
        depth=12,
        vlm_dim=256,
    )
    zero_count = assert_zero_init(expert)
    assert zero_count == 2 * 12 + 2 * 2

    objective = ResWAMFlowObjective(
        expert,
        world_loss_weight=0.1,
        action_tau_shift=1.0,
        world_tau_shift=5.0,
        solver_steps=10,
    )
    batch_size = 2
    actions = torch.randn(batch_size, 32, 14)
    action_mask = torch.ones(batch_size, 32, dtype=torch.bool)
    cur_dino = torch.randn(batch_size, TOKENS_PER_VIEW, DINO_DIM)
    fut_dino = torch.randn(batch_size, TOKENS_PER_VIEW, DINO_DIM)
    target = objective.residual(cur_dino, fut_dino)
    vlm_ctx = torch.randn(batch_size, 12, 256)
    vlm_mask = torch.ones(batch_size, 12, dtype=torch.bool)
    state = torch.randn(batch_size, 14)
    state_mask = torch.ones(batch_size, 14, dtype=torch.bool)
    gen = torch.Generator().manual_seed(123)

    # At initialization, AdaLN-Zero + FinalLayer zero-init guarantees exact zero velocity output
    out = objective(
        actions=actions,
        action_mask=action_mask,
        primary_target=target,
        current_primary=cur_dino,
        vlm_context=vlm_ctx,
        vlm_mask=vlm_mask,
        state=state,
        state_mask=state_mask,
        generator=gen,
    )
    assert torch.isfinite(out.loss)
    assert torch.count_nonzero(out.prediction.action) == 0
    assert torch.count_nonzero(out.prediction.primary) == 0

    sampled_actions, sampled_world, nfe = synchronous_sample(
        expert,
        current_primary=cur_dino,
        vlm_context=vlm_ctx,
        vlm_mask=vlm_mask,
        state=state,
        state_mask=state_mask,
        seeds=[101, 202],
    )
    assert nfe == 10
    assert sampled_actions.shape == (batch_size, 32, 14)
    assert sampled_world.shape == (batch_size, TOKENS_PER_VIEW, DINO_DIM)

    # Test full ResWAMSystem in LIBERO (11-D state, 0.3 dropout, H=8) online & cached modes
    sys_cfg = system_config(
        action_dim=7,
        state_dim=11,
        horizon=8,
        views=("head", "left_wrist"),
        world_views=("head",),
        world_loss_weight=0.5,
        state_dropout_prob=0.3,
    )
    sys_cfg["model"]["expert"] = {
        "hidden_dim": 256,
        "heads": 8,
        "ffn_dim": 512,
        "depth": 12,
        "variant": "bidirectional",
        "use_state_null_tokens": True,
        "layout": "libero",
    }
    system = ResWAMSystem(sys_cfg, vlm=DummyVLM(hidden_size=2048), dino=DummyDINO())
    prepare_fp32_trainable_parameters(system)

    vlm_ids = {id(p) for p in system.vlm.parameters()}
    vlm_params = [p for p in system.parameters() if p.requires_grad and id(p) in vlm_ids]
    expert_params = [p for p in system.parameters() if p.requires_grad and id(p) not in vlm_ids]
    optimizer = build_optimizer(vlm_params, expert_params)
    scheduler = WarmupCosine(optimizer, warmup=10, total=100)

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    cam_tensor = preprocess_camera(torch.from_numpy(dummy_img).permute(2, 0, 1))
    assert cam_tensor.shape == (3, 256, 256)

    online_batch: dict[str, Any] = {
        "current_cameras": {
            "head": torch.rand(batch_size, 3, 256, 256),
            "left_wrist": torch.rand(batch_size, 3, 256, 256),
        },
        "endpoint_cameras": {
            "head": torch.rand(batch_size, 3, 256, 256),
        },
        "qwen_cameras": {
            "head": [dummy_img, dummy_img],
            "left_wrist": [dummy_img, dummy_img],
        },
        "instructions": ["pick up the bowl", "place the mug"],
        "actions": torch.randn(batch_size, 8, 7),
        "action_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        "state": torch.randn(batch_size, 11),
        "state_mask": torch.ones(batch_size, 11, dtype=torch.bool),
    }
    res_online = system(online_batch, seed=7)
    assert res_online.loss.dtype == torch.float32 and torch.isfinite(res_online.loss)
    res_online.loss.backward()
    lrs = scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    assert "lr/vlm" in lrs and "lr/expert" in lrs

    cached_batch = {
        **online_batch,
        "dino_current": {"head": float_view(torch.randn(batch_size, 64, 768, dtype=torch.float32))},
        "dino_endpoint": {"head": float_view(torch.randn(batch_size, 64, 768, dtype=torch.float32))},
    }
    res_cached = system(cached_batch, seed=8)
    assert res_cached.loss.dtype == torch.float32 and torch.isfinite(res_cached.loss)

    system.eval()
    pred_actions = system.sample_actions(
        cameras={"head": [dummy_img, dummy_img], "left_wrist": [dummy_img, dummy_img]},
        instructions=["pick up the bowl", "place the mug"],
        normalized_state=torch.randn(batch_size, 11),
        seeds=[1, 2],
    )
    assert pred_actions.shape == (batch_size, 8, 7)


def test_released_checkpoint_layouts() -> None:
    """Parameter sets of the three released checkpoints (names, shapes and order)."""
    common = dict(dim=64, heads=4, ffn_dim=128, vlm_dim=64)
    libero = ResWAMExpert(
        action_dim=7, state_dim=11, horizon=8, use_state_null_tokens=True, layout="libero", **common
    )
    assert list(libero._parameters) == ["action_position", "view_identity", "state_position", "state_null"]
    assert tuple(libero.view_identity.shape) == (1, 1, 64)
    assert list(libero._modules)[-3:] == ["state_input", "action_state", "world_state"]
    robotwin = ResWAMExpert(action_dim=14, state_dim=14, horizon=32, **common)
    assert not hasattr(robotwin, "view_identity") and not hasattr(robotwin, "state_null")
    keys = set(robotwin.state_dict())
    robodojo_cfg = load_config(REPO_ROOT / "configs/robodojo.yaml")
    assert robodojo_cfg["model"]["expert"]["use_state_null_tokens"] is False
    robodojo = ResWAMExpert(
        action_dim=14,
        state_dim=14,
        horizon=32,
        use_state_null_tokens=bool(robodojo_cfg["model"]["expert"]["use_state_null_tokens"]),
        layout=robodojo_cfg["model"]["expert"]["layout"],
        **common,
    )
    assert set(robodojo.state_dict()) == keys
    libero_cfg = system_config(
        action_dim=7, state_dim=11, horizon=8, views=("head", "left_wrist"),
        state_dropout_prob=0.3, prompt_style="libero_cot",
    )
    assert libero_cfg["model"]["expert"]["layout"] == "libero"


def test_constant_floor_continuation_schedule() -> None:
    vlm, expert = [nn.Parameter(torch.zeros(2))], [nn.Parameter(torch.zeros(2))]
    optimizer = build_optimizer(vlm, expert)
    cosine = WarmupCosine(optimizer, warmup=10, total=200)
    for _ in range(200):
        cosine.step()
    lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
    floor = {g["name"]: g["min_lr"] for g in optimizer.param_groups}
    assert all(abs(lrs[k] - floor[k]) < 1e-15 for k in lrs)
    constant = ConstantFloor(optimizer, start=200, total=400)
    constant.load_state_dict(cosine.state_dict())  # scheduler migration at the stage boundary
    for _ in range(5):
        values = constant.step()
    assert constant.step_count == 205
    assert values == {"lr/vlm": 1e-6, "lr/expert": 1e-5}
    restored = ConstantFloor(optimizer, start=200, total=400)
    restored.load_state_dict(constant.state_dict())
    assert restored.step_count == 205


def test_official_rollout_seeds() -> None:
    import hashlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_robotwin2", REPO_ROOT / "scripts/eval/eval_robotwin2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = module.derive_rollout_seed(
        task="adjust_bottle", split="clean", episode_index=3, base_seed=0, chunk_index=5
    )
    identity = "\0".join(("adjust_bottle", "clean", "3", "0", "5")).encode()
    assert seed == int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


if __name__ == "__main__":
    test_configs_load_and_expand()
    test_released_checkpoint_layouts()
    test_constant_floor_continuation_schedule()
    test_official_rollout_seeds()
    test_codecs_and_state_featurization()
    test_cleaning_bspline_and_pi05_augmentation()
    test_expert_objective_and_system_online_and_cached()
    print(f"[test_smoke] All tests passed for reswam v{reswam.__version__}")

import os
import sys
from types import SimpleNamespace

import pytest
import torch


def _ensure_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(project_root, "src")
    vendor = os.path.join(project_root, "third_party", "tdmpc2", "tdmpc2")
    for p in (project_root, src_dir, vendor):
        if p not in sys.path:
            sys.path.insert(0, p)


def _tiny_cfg_cuda():
    return SimpleNamespace(
        action_dim=2,
        action_dims=[2],
        obs_shape={"state": (4,)},
        obs="state",
        task_dim=0,
        latent_dim=8,
        mlp_dim=16,
        num_bins=1,
        num_q=1,
        dropout=0.0,
        simnorm_dim=8,
        num_enc_layers=1,
        enc_dim=16,
        num_channels=16,
        log_std_min=-5.0,
        log_std_max=2.0,
        horizon=2,
        iterations=1,
        num_samples=4,
        num_pi_trajs=0,
        num_elites=2,
        temperature=1.0,
        min_std=0.05,
        max_std=0.5,
        mpc=False,
        lr=1e-3,
        enc_lr_scale=1.0,
        episodic=False,
        multitask=False,
        tasks=[],
        discount_denom=5,
        discount_min=0.95,
        discount_max=0.995,
        episode_length=5,
        tau=0.01,
        compile=False,
    )


def test_tdmpc2_gpu_train_and_infer():
    if os.environ.get("RUN_GPU_SMOKE") != "1":
        pytest.skip("Set RUN_GPU_SMOKE=1 to run GPU smoke")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    _ensure_paths()
    import third_party.tdmpc2.tdmpc2.tdmpc2 as td_mod

    cfg = _tiny_cfg_cuda()
    # Ensure module uses cuda:0 (default in upstream code)
    agent = td_mod.TDMPC2(cfg)

    # Inference on GPU
    obs = torch.zeros(cfg.obs_shape["state"], dtype=torch.float32, device="cuda")
    action = agent.act(obs, eval_mode=True)
    assert action.device.type == "cpu" or action.device.type == "cuda"  # act returns cpu copy
    print("tdmpc2 gpu action:", action)

    # One tiny training step on GPU
    obs_b = obs.unsqueeze(0)
    z = agent.model.encode(obs_b, task=None)
    zero_action = torch.zeros((1, cfg.action_dim), device="cuda")
    loss = agent.model.reward(z, zero_action, task=None).mean()
    agent.optim.zero_grad()
    loss.backward()
    agent.optim.step()
    print("tdmpc2 gpu loss:", float(loss.detach().cpu()))

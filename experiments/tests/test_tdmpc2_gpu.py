import os
import sys
import warnings
from types import SimpleNamespace

import pytest
import torch


def _ensure_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(project_root, "src")
    vendor = os.path.join(project_root, "third_party", "tdmpc2", "tdmpc2")
    for path in (project_root, src_dir, vendor):
        if path not in sys.path:
            sys.path.insert(0, path)


def _ensure_buffer_class():
    if getattr(torch.nn, "Buffer", None) is None:
        torch.nn.Buffer = torch.nn.Parameter  # type: ignore[attr-defined]


def _skip_if_gpu_not_ready():
    if os.environ.get("RUN_GPU_SMOKE") != "1":
        pytest.skip("Set RUN_GPU_SMOKE=1 to run GPU smoke")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not compatible with the current PyTorch installation.*",
            category=UserWarning,
        )
        major, minor = torch.cuda.get_device_capability(0)

    compiled_arches = set()
    for arch in torch.cuda.get_arch_list():
        try:
            _, code = arch.split("_")
            compiled_arches.add(divmod(int(code), 10))
        except ValueError:
            continue

    if compiled_arches and (major, minor) not in compiled_arches:
        pytest.skip(
            f"GPU compute capability sm_{major}{minor} not in compiled arches {sorted(compiled_arches)}"
        )
    return torch.device("cuda:0")


def _tiny_cfg():
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
        batch_size=2,
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
        episode_lengths=[5],
        tau=0.01,
        rho=0.5,
        reward_coef=1.0,
        value_coef=1.0,
        termination_coef=1.0,
        consistency_coef=1.0,
        entropy_coef=1e-4,
        grad_clip_norm=10.0,
        compile=False,
    )


@pytest.fixture(scope="module")
def tdmpc2_agent():
    device = _skip_if_gpu_not_ready()
    _ensure_buffer_class()
    _ensure_paths()
    import third_party.tdmpc2.tdmpc2.tdmpc2 as td_mod

    cfg = _tiny_cfg()
    agent = td_mod.TDMPC2(cfg)
    return agent, cfg, device


def test_tdmpc2_act_and_update(tdmpc2_agent):
    agent, cfg, device = tdmpc2_agent

    obs = torch.zeros(cfg.obs_shape["state"], dtype=torch.float32, device=device)
    action = agent.act(obs, eval_mode=True)
    assert action.shape == (cfg.action_dim,)
    assert torch.all(torch.isfinite(action))

    obs_b = obs.unsqueeze(0)
    z = agent.model.encode(obs_b, task=None)
    zero_action = torch.zeros((1, cfg.action_dim), device=device)
    loss = agent.model.reward(z, zero_action, task=None).mean()
    agent.optim.zero_grad()
    loss.backward()
    agent.optim.step()

    class _FakeBuffer:
        def __init__(self, cfg, device):
            T, B = cfg.horizon, cfg.batch_size
            self.obs = torch.zeros((T + 1, B, cfg.obs_shape["state"][0]), device=device)
            self.action = torch.zeros((T, B, cfg.action_dim), device=device)
            self.reward = torch.zeros((T, B, 1), device=device)
            self.terminated = torch.zeros((T, B, 1), device=device)

        def sample(self):
            return self.obs, self.action, self.reward, self.terminated, None

    stats = agent.update(_FakeBuffer(cfg, device))
    assert hasattr(stats, "keys")
    assert {"total_loss", "grad_norm", "pi_loss"}.issubset(set(stats.keys()))

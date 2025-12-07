import os
import sys

import numpy as np
import pytest
import ruamel.yaml as yaml


def _ensure_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(project_root, "src")
    vendor = os.path.join(project_root, "third_party", "dreamerv3")
    for p in (project_root, src_dir, vendor):
        if p not in sys.path:
            sys.path.insert(0, p)
    return project_root


def _load_config(cfg_path):
    y = yaml.YAML(typ="safe")
    with open(cfg_path, "r") as f:
        return y.load(f)


def _tiny_dv3_config(configs):
    import elements

    cfg = elements.Config(configs["defaults"])
    # Shrink model sizes for a quick smoke
    cfg = cfg.update(
        batch_size=2,
        batch_length=8,
        report_length=4,
        run=dict(envs=1, eval_envs=0, train_ratio=1, steps=32, log_every=16, report_every=16),
        jax=dict(platform="cuda", compute_dtype="float32", prealloc=False, jit=True),
    )
    cfg = cfg.update(
        dyn=dict(rssm=dict(deter=128, hidden=64, stoch=8, classes=4, blocks=2)),
        enc=dict(simple=dict(depth=16, mults=[1, 1, 1, 1], layers=1, units=64)),
        dec=dict(simple=dict(depth=16, mults=[1, 1, 1, 1], layers=1, units=64)),
    )
    return cfg


def _dummy_spaces():
    import elements

    obs_space = {
        "image": elements.Space(np.uint8, (64, 64, 3)),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
        "reward": elements.Space(np.float32, ()),
    }
    act_space = {"action": elements.Space(np.int32, (), 0, 2)}
    return obs_space, act_space


def test_dreamerv3_heavy_gpu_policy_and_update():
    if os.environ.get("RUN_GPU_SMOKE") != "1" or os.environ.get("RUN_DV3_HEAVY") != "1":
        pytest.skip("Set RUN_GPU_SMOKE=1 and RUN_DV3_HEAVY=1 to run DreamerV3 heavy GPU smoke")

    _ensure_paths()
    try:
        import jax  # noqa: F401
        import elements
        from dreamerv3.agent import Agent
    except Exception as exc:  # pragma: no cover - optional deps
        pytest.skip(f"DreamerV3 deps missing: {exc}")

    configs = _load_config(os.path.join(_ensure_paths(), "third_party", "dreamerv3", "dreamerv3", "configs.yaml"))
    cfg = _tiny_dv3_config(configs)

    obs_space, act_space = _dummy_spaces()
    agent = Agent(obs_space, act_space, cfg)

    # Prepare dummy batch
    B = cfg.batch_size
    obs = {
        "image": np.zeros((B, 64, 64, 3), np.uint8),
        "is_first": np.zeros((B,), bool),
        "is_last": np.zeros((B,), bool),
        "is_terminal": np.zeros((B,), bool),
        "reward": np.zeros((B,), np.float32),
    }

    carry = agent.init_policy(B)
    carry, action = agent.policy(carry, obs, mode="train")
    print("dreamerv3 gpu policy action keys:", action.keys())

    # Minimal train/init step
    train_carry = agent.init_train(B)
    _ = agent.train(train_carry, obs)
    print("dreamerv3 gpu train step completed")

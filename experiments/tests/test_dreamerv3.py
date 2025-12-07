import os
import sys
import pytest

jax = pytest.importorskip("jax")  # skip if optional deps missing


def _ensure_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(project_root, "src")
    for p in (project_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


def test_dreamerv3_module_imports():
    _ensure_paths()
    mod = __import__("models.dreamerv3", fromlist=["DreamerV3Agent"])
    assert hasattr(mod, "DreamerV3Agent")

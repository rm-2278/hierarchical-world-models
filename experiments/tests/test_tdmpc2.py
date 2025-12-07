import os
import sys


def _ensure_paths():
    """Add project root and src to sys.path for test imports."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(project_root, "src")
    for p in (project_root, src_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


def test_tdmpc2_module_imports():
    _ensure_paths()
    mod = __import__("models.tdmpc2", fromlist=["TDMPC2Agent"])
    assert hasattr(mod, "TDMPC2Agent")

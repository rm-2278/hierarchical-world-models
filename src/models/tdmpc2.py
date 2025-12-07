import importlib
import sys
from pathlib import Path
from typing import Any

# Make vendor package imports (tdmpc2/common/...) resolvable.
VENDOR_PKG = Path(__file__).resolve().parents[2] / "third_party" / "tdmpc2" / "tdmpc2"
if str(VENDOR_PKG) not in sys.path:
    sys.path.insert(0, str(VENDOR_PKG))


class TDMPC2Agent:
    """Thin wrapper around third_party.tdmpc2.TDMPC2.

    The upstream class expects a config namespace with fields documented in
    third_party/tdmpc2/tdmpc2/config.yaml. We keep forwarding minimal methods.
    """

    def __init__(self, config: Any):
        try:
            module = importlib.import_module("tdmpc2")
        except ImportError as exc:  # pragma: no cover - optional dep surface
            missing = "Ensure torch + tdmpc2 deps are installed"
            raise ImportError(f"tdmpc2 import failed: {exc}. {missing}") from exc
        AgentCls = getattr(module, "TDMPC2")
        self._agent = AgentCls(config)

    def act(self, obs, **kwargs):
        return self._agent.act(obs, **kwargs)

    def update(self, buffer):
        return self._agent.update(buffer)

    def __getattr__(self, name: str):
        # Delegate everything else to the wrapped agent (save/load/etc.).
        return getattr(self._agent, name)

import importlib
import sys
from pathlib import Path
from typing import Any

# Ensure vendor path is importable.
VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "dreamerv3"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


class DreamerV3Agent:
    """Thin wrapper around third_party.dreamerv3 Agent.

    The upstream Agent expects (obs_space, act_space, config).
    We do not enforce shapes here; caller passes the correct spaces/config.
    """

    def __init__(self, obs_space: Any, act_space: Any, config: Any):
        try:
            module = importlib.import_module("dreamerv3.agent")
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            missing = (
                "Install JAX + DreamerV3 deps (see third_party/dreamerv3/requirements.txt)"
            )
            raise ImportError(f"dreamerv3 import failed: {exc}. {missing}") from exc
        AgentCls = getattr(module, "Agent")
        self._agent = AgentCls(obs_space, act_space, config)

    def __getattr__(self, name: str):
        # Delegate unknown attributes to the wrapped agent (policy, train, etc.).
        return getattr(self._agent, name)

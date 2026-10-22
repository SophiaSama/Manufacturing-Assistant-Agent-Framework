"""FastAPI backend server for NGA Manufacturing Assistant Web Interface.

Backward-compatibility shim re-exporting from enterprise_agent.ui.server.
"""

from __future__ import annotations

import sys
from typing import Any

import enterprise_agent.ui.server as _ea_server

# Re-export all public symbols
for _name in dir(_ea_server):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_ea_server, _name)


class _ServerShimModule(sys.modules[__name__].__class__):
    """Module class that forwards attribute mutations to enterprise_agent.ui.server."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(_ea_server, name):
            setattr(_ea_server, name, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(_ea_server, name)


sys.modules[__name__].__class__ = _ServerShimModule

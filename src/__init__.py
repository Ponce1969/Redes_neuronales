"""Top-level package for the neural_core project.

Expose legacy module names (``autograd``, ``core``, ``engine``) so that
scripts inserting ``src`` directly into ``sys.path`` continue to work without
import errors or duplicated module instances.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Iterable


def _alias_modules(src_package: str, public_name: str, submodules: Iterable[str]) -> None:
    """Register ``public_name`` as alias for ``src_package`` in ``sys.modules``."""

    module = import_module(src_package)
    sys.modules.setdefault(public_name, module)

    for sub in submodules:
        full_src = f"{src_package}.{sub}"
        full_public = f"{public_name}.{sub}"
        sys.modules.setdefault(full_public, import_module(full_src))


_alias_modules("src.autograd", "autograd", ("value", "functional", "ops"))
_alias_modules("src.core.autograd_numpy", "autograd_numpy", ("tensor", "loss"))
_alias_modules("src.core", "core", ("trm_block", "trm_act_block"))
_alias_modules("src.engine", "engine", ("trainer", "rl_trainer", "dataset", "predictor"))

__all__ = ["autograd", "core", "engine"]

"""Register FreeAct's task suites and configure the LIBERO runtime.

Importing this module registers the suites. Call ``prepare_suite_env`` in each
worker process before creating an environment to apply its scene configuration.
"""

import importlib.util
import os

import yaml


def _ensure_libero_config() -> None:
    """Write LIBERO's default ``config.yaml`` so that importing it never prompts interactively."""
    config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_file = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_file):
        return
    root = os.path.join(os.path.dirname(importlib.util.find_spec("libero").origin), "libero")
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(
            {
                "benchmark_root": root,
                "bddl_files": os.path.join(root, "bddl_files"),
                "init_states": os.path.join(root, "init_files"),
                "datasets": os.path.join(root, "../datasets"),
                "assets": os.path.join(root, "assets"),
            },
            f,
        )


_ensure_libero_config()

from .suites import SUITE_DEFS, prepare_suite_env, register_suites  # noqa: E402

register_suites()

__all__ = ["SUITE_DEFS", "prepare_suite_env", "register_suites"]

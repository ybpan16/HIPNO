"""Run the environment-configured v9-A training implementation."""

from __future__ import annotations

import argparse
import runpy


TRAINING_MODULE = "models.training_v9_tauA"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HIPNO v9-A training. Training settings are supplied through "
            "the checked-in shell configuration and PINN_* environment variables."
        )
    )
    return parser.parse_args()


def main() -> None:
    """Execute the canonical v9-A training module."""
    _parse_args()
    runpy.run_module(TRAINING_MODULE, run_name="__main__")


if __name__ == "__main__":
    main()

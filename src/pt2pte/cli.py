"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import export_pte


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert an exportable PyTorch checkpoint to an Ethos-U PTE"
    )
    parser.add_argument("--config", required=True, help="YAML pipeline configuration")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Override model.weights; relative paths are resolved from the current directory",
    )
    parser.add_argument(
        "--target",
        "--accelerator",
        dest="accelerator",
        help="Override target.accelerator, for example ethos-u85-256",
    )
    args = parser.parse_args(argv)
    pte = export_pte(
        load_config(args.config, weights=args.weights, accelerator=args.accelerator)
    )
    print(f"PTE: {pte}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Artifact manifests without content hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_manifest(
    path: str | Path,
    *,
    pte: str | Path,
    metadata: dict[str, Any],
) -> None:
    manifest_path = Path(path)
    pte_path = Path(pte).resolve()
    if not pte_path.is_file() or pte_path.stat().st_size == 0:
        raise FileNotFoundError(f"PTE file does not exist or is empty: {pte_path}")
    payload = dict(metadata)
    payload["pte"] = {"path": str(pte_path), "bytes": pte_path.stat().st_size}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

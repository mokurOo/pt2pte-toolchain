import json

import pt2pte.artifacts as artifacts_module


def test_write_manifest_records_file_size_without_hashes(tmp_path):
    pte = tmp_path / "model.pte"
    pte.write_bytes(b"pte")
    manifest = tmp_path / "manifest.json"

    write_manifest = getattr(artifacts_module, "write_manifest", None)
    assert callable(write_manifest)
    write_manifest(manifest, pte=pte, metadata={"target": "ethos-u85-256"})
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["pte"]["bytes"] == 3
    assert data["pte"]["path"] == str(pte.resolve())
    assert not any("sha" in key.lower() or "hash" in key.lower() for key in data["pte"])

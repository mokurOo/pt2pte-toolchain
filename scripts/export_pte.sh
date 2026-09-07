#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/export_pte.sh [CONFIG] [options]

Convert a configured PyTorch model to a PTE.

Options are forwarded to `python -m pt2pte`:
  --weights PATH       Override model.weights
  --target NAME        Override target.accelerator
  --accelerator NAME   Alias for --target
EOF
  exit 0
fi

toolkit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
  config_path="$1"
  shift
else
  config_path="${toolkit_root}/configs/yolo_hand_pose.yaml"
fi
executorch_root="${EXECUTORCH_ROOT:-${toolkit_root}/third_party/executorch}"
python_bin="${PT2PTE_PYTHON:-${toolkit_root}/.venv/bin/python}"
cmsis_nn_path="${PT2PTE_CMSIS_NN_PATH:-${toolkit_root}/build/cmsis/backends/cortex_m/cmsis_nn-build}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing toolkit environment. Run scripts/bootstrap.sh first." >&2
  exit 1
fi
if [[ ! -d "${executorch_root}/executorch" && ! -d "${executorch_root}/src/executorch" ]]; then
  echo "Invalid EXECUTORCH_ROOT: ${executorch_root}" >&2
  exit 1
fi
if ! find "${cmsis_nn_path}" -maxdepth 1 -type f -name 'cmsis_nn*.so' -print -quit 2>/dev/null | grep -q .; then
  echo "Missing CMSIS-NN Python binding: ${cmsis_nn_path}" >&2
  echo "Run scripts/bootstrap.sh or set PT2PTE_CMSIS_NN_PATH." >&2
  exit 1
fi

python_site="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${toolkit_root}/src:${cmsis_nn_path}:${executorch_root}/src:${executorch_root}:${python_site}${PYTHONPATH:+:${PYTHONPATH}}"

# A mixed torch/torchao environment produces opaque torch.export failures such
# as `call_module _guards_fn`. Fail early with the actual module origins.
PT2PTE_EXPECTED_SITE="${python_site}" \
  PT2PTE_EXPECTED_TORCH_PREFIX="${PT2PTE_EXPECTED_TORCH_PREFIX:-2.13.}" \
  "${python_bin}" - <<'PY'
import os
from pathlib import Path

import torch
import torchao

expected_site = Path(os.environ["PT2PTE_EXPECTED_SITE"]).resolve()
for name, module in (("torch", torch), ("torchao", torchao)):
    origin = Path(module.__file__).resolve()
    if expected_site not in origin.parents:
        raise SystemExit(
            f"{name} is loaded from {origin}, outside the selected environment "
            f"{expected_site}. Remove that package from PYTHONPATH or select a "
            "matching PT2PTE_PYTHON."
        )

expected_prefix = os.environ.get("PT2PTE_EXPECTED_TORCH_PREFIX", "")
if expected_prefix and not torch.__version__.startswith(expected_prefix):
    raise SystemExit(
        f"Unsupported torch version {torch.__version__}; this ExecuTorch v1.4.1 "
        f"toolchain expects {expected_prefix}*. Set "
        "PT2PTE_EXPECTED_TORCH_PREFIX= to override deliberately."
    )

print(f"torch: {torch.__version__} ({torch.__file__})")
print(f"torchao: {getattr(torchao, '__version__', 'unknown')} ({torchao.__file__})")
PY

exec "${python_bin}" -m pt2pte --config "${config_path}" "$@"

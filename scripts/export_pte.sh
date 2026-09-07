#!/usr/bin/env bash
set -euo pipefail

toolkit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${1:-${toolkit_root}/configs/yolo_hand_pose.yaml}"
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
export PYTHONPATH="${toolkit_root}/src:${cmsis_nn_path}:${executorch_root}/src:${executorch_root}:${python_site}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" -m pt2pte --config "${config_path}"

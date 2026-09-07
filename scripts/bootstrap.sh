#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [options]

Install the PT-to-PTE host toolchain.

Options:
  -d, --device MODE       cpu (default) or gpu
      --torch-version V   CPU torch version, default: 2.13.0
      --torch-index-url U CPU torch index, default: PyTorch test CPU index
  -h, --help              Show this help

Environment equivalents:
  PT2PTE_DEVICE=cpu|gpu
  PT2PTE_TORCH_VERSION=2.13.0
  PT2PTE_TORCH_INDEX_URL=https://download.pytorch.org/whl/test/cpu
EOF
}

device_mode="${PT2PTE_DEVICE:-cpu}"
torch_version="${PT2PTE_TORCH_VERSION:-2.13.0}"
torch_index_url="${PT2PTE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/test/cpu}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--device)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      device_mode="$2"
      shift 2
      ;;
    --torch-version)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      torch_version="$2"
      shift 2
      ;;
    --torch-index-url)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      torch_index_url="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${device_mode}" in
  cpu|gpu) ;;
  *)
    echo "Invalid device mode '${device_mode}'; expected cpu or gpu." >&2
    exit 2
    ;;
esac

if [[ "${device_mode}" == "gpu" ]] && ! command -v nvcc >/dev/null 2>&1; then
  echo "GPU mode requires a CUDA toolkit with nvcc on PATH." >&2
  echo "Use --device cpu on this machine, or install/configure CUDA first." >&2
  exit 1
fi

toolkit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${toolkit_root}/.venv"
executorch_root="${toolkit_root}/third_party/executorch"
executorch_tag="v1.4.1"

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ "${device_mode}" == "cpu" ]]; then
  cpu_torch_version="${torch_version}"
  if [[ "${cpu_torch_version}" != *+cpu ]]; then
    cpu_torch_version="${cpu_torch_version}+cpu"
  fi
  echo "Installing CPU PyTorch torch==${cpu_torch_version} from ${torch_index_url}"
  python -m pip install \
    --index-url "${torch_index_url}" \
    --extra-index-url https://pypi.org/simple \
    "torch==${cpu_torch_version}"

  # ExecuTorch probes nvcc while resolving its torch/torchao wheels. Keep that
  # probe on the CPU path even when a CUDA toolkit happens to be installed.
  path_without_nvcc=()
  IFS=: read -r -a path_entries <<< "${PATH}"
  for path_entry in "${path_entries[@]}"; do
    if [[ -x "${path_entry}/nvcc" ]]; then
      continue
    fi
    path_without_nvcc+=("${path_entry}")
  done
  path_for_cpu_install="$(IFS=:; printf '%s' "${path_without_nvcc[*]}")"

  cmake_args_value="${CMAKE_ARGS:-}"
  cmake_args_value="${cmake_args_value//-DEXECUTORCH_BUILD_CUDA=ON/-DEXECUTORCH_BUILD_CUDA=OFF}"
  if [[ "${cmake_args_value}" != *-DEXECUTORCH_BUILD_CUDA=OFF* ]]; then
    cmake_args_value="${cmake_args_value}${cmake_args_value:+ }-DEXECUTORCH_BUILD_CUDA=OFF"
  fi
  export CMAKE_ARGS="${cmake_args_value}"
fi

if [[ ! -d "${executorch_root}/.git" ]]; then
  mkdir -p "${toolkit_root}/third_party"
  git clone --branch "${executorch_tag}" --depth 1 --recurse-submodules \
    https://github.com/pytorch/executorch.git "${executorch_root}"
fi

cd "${executorch_root}"
if [[ "${device_mode}" == "cpu" ]]; then
  PATH="${path_for_cpu_install}" DEBUG=0 ./install_executorch.sh \
    --use-pt-pinned-commit \
    --editable \
    --minimal
else
  DEBUG=0 ./install_executorch.sh \
    --use-pt-pinned-commit \
    --editable \
    --minimal
fi

python -m pip install \
  ai-edge-model-explorer==0.1.33 \
  cmake==3.31.10 \
  ml_dtypes==0.5.1 \
  ninja==1.13.2 \
  pybind11==3.1.0 \
  ethos-u-vela==5.1.0 \
  pte-adapter-model-explorer==0.0.2
python -m pip install --no-dependencies \
  flatbuffers==24.3.25 \
  tosa-tools==2026.5.0 \
  tosa-adapter-model-explorer==0.1.0
python -m pip install -e "${toolkit_root}"

cmake -S "${executorch_root}" -B "${toolkit_root}/build/cmsis" \
  -DEXECUTORCH_BUILD_CMSIS_NN_PYBINDS=ON \
  -DEXECUTORCH_BUILD_CORTEX_M=ON \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_CUDA=OFF
cmake --build "${toolkit_root}/build/cmsis" --target cmsis_nn --parallel 2

python -c "import torch, torchao, ultralytics; print(torch.__version__, torchao.__version__, ultralytics.__version__)"
PYTHONPATH="${toolkit_root}/build/cmsis/backends/cortex_m/cmsis_nn-build" \
  python -c "import cmsis_nn; print(cmsis_nn.__file__)"
vela --version

#!/usr/bin/env bash
set -euo pipefail

toolkit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${toolkit_root}/.venv"
executorch_root="${toolkit_root}/third_party/executorch"
executorch_tag="v1.4.1"

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ ! -d "${executorch_root}/.git" ]]; then
  mkdir -p "${toolkit_root}/third_party"
  git clone --branch "${executorch_tag}" --depth 1 --recurse-submodules \
    https://github.com/pytorch/executorch.git "${executorch_root}"
fi

cd "${executorch_root}"
DEBUG=0 ./install_executorch.sh \
  --use-pt-pinned-commit \
  --editable \
  --minimal

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
  -DEXECUTORCH_BUILD_TESTS=OFF
cmake --build "${toolkit_root}/build/cmsis" --target cmsis_nn --parallel 2

python -c "import torch, torchao, ultralytics; print(torch.__version__, torchao.__version__, ultralytics.__version__)"
PYTHONPATH="${toolkit_root}/build/cmsis/backends/cortex_m/cmsis_nn-build" \
  python -c "import cmsis_nn; print(cmsis_nn.__file__)"
vela --version

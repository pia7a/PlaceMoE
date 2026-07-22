#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
source "${ASCEND_INSTALL_PATH}/bin/setenv.bash"
export CMAKE_PREFIX_PATH="${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake:${CMAKE_PREFIX_PATH:-}"

cmake -S "${BASE_DIR}/csrc/opp" -B "${BASE_DIR}/build/opp"
cmake --build "${BASE_DIR}/build/opp" --target binary --clean-first -j "${BUILD_JOBS:-8}"
cmake --build "${BASE_DIR}/build/opp" --target package -j "${BUILD_JOBS:-8}"
cmake --install "${BASE_DIR}/build/opp" --prefix "${BASE_DIR}/build/opp_install"

export ASCEND_CUSTOM_OPP_PATH="${BASE_DIR}/build/opp_install/packages/vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}"
cd "${BASE_DIR}"
python setup.py build_ext --inplace --force

printf 'Built HierMoE NPU planner ops. ASCEND_CUSTOM_OPP_PATH=%s\n' "${ASCEND_CUSTOM_OPP_PATH}"

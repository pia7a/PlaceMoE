#!/usr/bin/env bash
# Stage, verify, and promote EP32 artifacts on all worker nodes.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 ARTIFACT..." >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../../.." && pwd)
# shellcheck source=common.sh
source "${script_dir}/common.sh"
# shellcheck source=ssh.sh
source "${script_dir}/ssh.sh"
repro_configure_ssh "${script_dir}"

destination=${PLACEMOE_REPRO_REMOTE_REPO_ROOT:-${repo_root}}/results
generation="placemoe_$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}"
remote_specs=("${repro_remote_specs[@]}")
artifacts=("$@")
names=()
checksums=()
for artifact in "${artifacts[@]}"; do
  if [[ ! -s "${artifact}" ]]; then
    echo "missing publication artifact: ${artifact}" >&2
    exit 1
  fi
  names+=("$(basename "${artifact}")")
  checksum=$(sha256sum "${artifact}")
  checksums+=("${checksum%% *}")
done

# Phase 1: every node receives and verifies the complete generation while the
# live files remain untouched.
for spec in "${remote_specs[@]}"; do
  port=${spec%%:*}
  stage="${destination}/.${generation}_${port}"
  repro_ssh "${port}" "mkdir -p '${stage}'"
  repro_scp_to "${port}" "${stage}/" "${artifacts[@]}"

  for index in "${!artifacts[@]}"; do
    actual=$(repro_ssh "${port}" "sha256sum '${stage}/${names[index]}'")
    actual=${actual%% *}
    if [[ "${actual}" != "${checksums[index]}" ]]; then
      echo "artifact checksum mismatch on port ${port}: ${names[index]}" >&2
      exit 1
    fi
  done
done

# Phase 2: promotion only begins after all nodes have the verified generation.
for spec in "${remote_specs[@]}"; do
  port=${spec%%:*}
  stage="${destination}/.${generation}_${port}"
  promotion="set -e; mkdir -p '${destination}';"
  for name in "${names[@]}"; do
    promotion+=" mv -f '${stage}/${name}' '${destination}/${name}';"
  done
  promotion+=" rmdir '${stage}'"
  repro_ssh "${port}" "${promotion}"
done

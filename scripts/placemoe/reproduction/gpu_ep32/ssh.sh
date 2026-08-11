#!/usr/bin/env bash

# Shared SSH configuration for the four-node EP32 testbed.

repro_ssh_user=${PLACEMOE_REPRO_SSH_USER:-root}
repro_ssh_host=${PLACEMOE_REPRO_SSH_HOST:-}
repro_remote_specs=(
  "${PLACEMOE_REPRO_RANK1_PORT:-}:1"
  "${PLACEMOE_REPRO_RANK2_PORT:-}:2"
  "${PLACEMOE_REPRO_RANK3_PORT:-}:3"
)
repro_ssh_args=(
  -o StrictHostKeyChecking="${PLACEMOE_REPRO_SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
  -o ConnectTimeout="${PLACEMOE_REPRO_SSH_CONNECT_TIMEOUT:-15}"
)
repro_scp_args=(
  -o StrictHostKeyChecking="${PLACEMOE_REPRO_SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
  -o ConnectTimeout="${PLACEMOE_REPRO_SSH_CONNECT_TIMEOUT:-15}"
)

repro_configure_ssh() {
  local script_dir=$1
  local port rank spec
  if [[ -z "${repro_ssh_host}" || -z "${repro_ssh_user}" ]]; then
    echo "PLACEMOE_REPRO_SSH_HOST and PLACEMOE_REPRO_SSH_USER must be set in ${repro_cluster_config_path}" >&2
    return 2
  fi
  for spec in "${repro_remote_specs[@]}"; do
    port=${spec%%:*}
    rank=${spec##*:}
    if [[ ! "${port}" =~ ^[1-9][0-9]*$ ]] || ((port > 65535)); then
      echo "PLACEMOE_REPRO_RANK${rank}_PORT must be an integer in [1, 65535]" >&2
      return 2
    fi
  done
  if [[ -n "${PLACEMOE_REPRO_SSH_KNOWN_HOSTS_FILE:-}" ]]; then
    repro_ssh_args+=(-o "UserKnownHostsFile=${PLACEMOE_REPRO_SSH_KNOWN_HOSTS_FILE}")
    repro_scp_args+=(-o "UserKnownHostsFile=${PLACEMOE_REPRO_SSH_KNOWN_HOSTS_FILE}")
  fi
  if [[ -n "${PLACEMOE_REPRO_SSH_KEY:-}" ]]; then
    unset SSH_ASKPASS SSH_ASKPASS_REQUIRE
    if [[ ! -r "${PLACEMOE_REPRO_SSH_KEY}" ]]; then
      echo "PLACEMOE_REPRO_SSH_KEY is not readable: ${PLACEMOE_REPRO_SSH_KEY}" >&2
      return 1
    fi
    repro_ssh_args+=(-i "${PLACEMOE_REPRO_SSH_KEY}" -o BatchMode=yes)
    repro_scp_args+=(-i "${PLACEMOE_REPRO_SSH_KEY}" -o BatchMode=yes)
  elif [[ -n "${PLACEMOE_REPRO_SSH_PASSWORD:-}" ]]; then
    export PLACEMOE_REPRO_SSH_PASSWORD
    export SSH_ASKPASS="${script_dir}/askpass.sh"
    export SSH_ASKPASS_REQUIRE=force
    export DISPLAY=${DISPLAY:-:0}
  elif [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
    unset SSH_ASKPASS SSH_ASKPASS_REQUIRE
    repro_ssh_args+=(-o BatchMode=yes)
    repro_scp_args+=(-o BatchMode=yes)
  else
    echo "EP32 remote access requires PLACEMOE_REPRO_SSH_KEY, PLACEMOE_REPRO_SSH_PASSWORD, or SSH_AUTH_SOCK." >&2
    return 1
  fi
}

repro_ssh() {
  local port=$1
  shift
  setsid -w ssh "${repro_ssh_args[@]}" -p "${port}" "${repro_ssh_user}@${repro_ssh_host}" "$@"
}

repro_scp_from() {
  local port=$1
  local source=$2
  local destination=$3
  setsid -w scp "${repro_scp_args[@]}" -r -P "${port}" \
    "${repro_ssh_user}@${repro_ssh_host}:${source}" "${destination}"
}

repro_scp_to() {
  local port=$1
  local destination=$2
  shift 2
  setsid -w scp "${repro_scp_args[@]}" -P "${port}" "$@" \
    "${repro_ssh_user}@${repro_ssh_host}:${destination}"
}

#!/usr/bin/env bash

set -euo pipefail
: "${PLACEMOE_REPRO_SSH_PASSWORD:?PLACEMOE_REPRO_SSH_PASSWORD is required}"
printf '%s\n' "${PLACEMOE_REPRO_SSH_PASSWORD}"

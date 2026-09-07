#!/usr/bin/env bash
set -euo pipefail

toolkit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${toolkit_root}/scripts/export_pte.sh" "${toolkit_root}/configs/yolo_hand_pose.yaml"

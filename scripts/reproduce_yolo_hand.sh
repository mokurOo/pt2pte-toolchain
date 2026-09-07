#!/usr/bin/env bash
set -euo pipefail

toolkit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${PT2PTE_CONFIG:-${toolkit_root}/configs/yolo_hand_pose.yaml}"
export_args=()

usage() {
  cat <<'EOF'
Usage: scripts/reproduce_yolo_hand.sh [options]

Convert a YOLO pose checkpoint with the default pipeline configuration.

Options:
  -w, --weights PATH  Override model.weights in the configuration
  -t, --target NAME   Override target.accelerator, e.g. ethos-u85-256
  -c, --config PATH   Use another pipeline configuration
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--weights)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      export_args+=(--weights "$2")
      shift 2
      ;;
    -t|--target)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      export_args+=(--target "$2")
      shift 2
      ;;
    -c|--config)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      config_path="$2"
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

exec "${toolkit_root}/scripts/export_pte.sh" "${config_path}" "${export_args[@]}"

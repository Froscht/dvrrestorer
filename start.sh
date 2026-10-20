#!/usr/bin/env bash
# Start DVR Restorer GUI on Linux/macOS.
# Make executable with: chmod +x start.sh
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m dvrrestorer.gui.app "$@"

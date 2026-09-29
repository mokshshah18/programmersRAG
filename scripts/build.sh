#!/usr/bin/env bash
# Configure and build programmersRAG on Linux/macOS (GCC or Clang + Ninja).
# Usage: scripts/build.sh [preset]   presets: release (default), debug, core
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
preset="${1:-release}"
cmake --preset "$preset"
cmake --build --preset "$preset"

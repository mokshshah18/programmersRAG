#!/bin/bash

# Directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Always append to config next to the script
CONFIG_FILE="$SCRIPT_DIR/config"

echo "Scanning from: $(pwd)"
echo "Appending absolute paths to: $CONFIG_FILE"
echo ""

# Recursively append absolute paths as they are found
find . -type f \( \
    -name "*.cpp" -o \
    -name "*.h"   -o \
    -name "*.hpp" \
\) | while read -r f; do
    ABS_PATH="$(realpath "$f")"
    echo "$ABS_PATH" >> "$CONFIG_FILE"
    echo "Added: $ABS_PATH"
done

echo ""
echo "Done."

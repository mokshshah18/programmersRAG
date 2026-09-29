#!/bin/bash

CONFIG_FILE="./config"
OUTPUT_FILE="./cleaned_config"
MAX_SIZE=60000  # 60 KB

# Clear output file if it exists
> "$OUTPUT_FILE"

# Read each path from config
while IFS= read -r FILE_PATH; do
    # Skip empty lines
    [ -z "$FILE_PATH" ] && continue

    # Skip if path contains forbidden patterns (case-insensitive)
    if echo "$FILE_PATH" | grep -qiE "/build/|/install/|test|medicar"; then
        continue
    fi

    # Skip if file does not exist
    [ -f "$FILE_PATH" ] || continue

    # Check file size
    FILE_SIZE=$(stat -c%s "$FILE_PATH")
    if [ "$FILE_SIZE" -ge "$MAX_SIZE" ]; then
        continue
    fi

    # Path passes all checks; write to output
    echo "$FILE_PATH" >> "$OUTPUT_FILE"
done < "$CONFIG_FILE"

echo "Cleaned config paths written to $OUTPUT_FILE"

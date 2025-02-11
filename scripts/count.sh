#!/bin/bash

# usage : main_exps_ckpt8k/count.sh main_exps_ckpt8k
TARGET_DIR=${1:-.}

for SUBDIR in "$TARGET_DIR"/*/; do
    if [ -d "$SUBDIR" ]; then
        FILE_COUNT=$(find "$SUBDIR" -type f | wc -l)
        echo "Folder: $(basename "$SUBDIR") - Files: $FILE_COUNT"
    fi
done

#!/bin/bash
# Run evaluation on both vanilla and semantic models

cd "$(dirname "$0")"
python3 test_both_models.py "$@"

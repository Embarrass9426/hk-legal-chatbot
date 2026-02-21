#!/bin/bash
# Wrapper to run python commands with TensorRT libraries in LD_LIBRARY_PATH
# Usage: ./backend/run_trt_wsl.sh <command> [args...]

# Determine script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Path to TensorRT libs inside the virtual environment
TRT_LIB_PATH="$PROJECT_ROOT/.wsl_venv/lib/python3.12/site-packages/tensorrt_libs"

if [ -d "$TRT_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$TRT_LIB_PATH:$LD_LIBRARY_PATH"
    # echo "Added $TRT_LIB_PATH to LD_LIBRARY_PATH"
else
    echo "Warning: TensorRT libs not found at $TRT_LIB_PATH"
fi

# Activate venv if not already
if [[ "$VIRTUAL_ENV" == "" ]]; then
    source "$PROJECT_ROOT/.wsl_venv/bin/activate"
fi

# Execute the command
exec "$@"

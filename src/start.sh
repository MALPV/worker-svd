#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Add ComfyUI to Python path
export PYTHONPATH="/ComfyUI:${PYTHONPATH}"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python3 -u -m ComfyUI.src.handler --rp_serve_api --rp_api_host=0.0.0.0
else
    python3 -u -m ComfyUI.src.handler
fi
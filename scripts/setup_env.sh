#!/bin/bash
# Source this before running any Isaac Sim command with GUI
# Usage: source scripts/setup_env.sh

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate isaac_lab

# Fix Omniverse native library paths (required for GUI mode)
OMNI_BASE="$HOME/miniconda3/envs/isaac_lab/lib/python3.10/site-packages"
KIT_EXT="$HOME/.local/share/ov/data/exts/v2"
SO_DIRS=$(find "$OMNI_BASE/omni" "$KIT_EXT" -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${SO_DIRS}${OMNI_BASE}/omni:${OMNI_BASE}/omni/kernel:$LD_LIBRARY_PATH"

# Accept NVIDIA Omniverse EULA
export OMNI_KIT_ACCEPT_EULA=YES

echo "[setup_env] Conda: isaac_lab | LD_LIBRARY_PATH: $(echo $LD_LIBRARY_PATH | tr ':' '\n' | wc -l) dirs | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

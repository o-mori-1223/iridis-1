#!/bin/bash
# Structural color simulator launcher -- FDTD backend (WSL / s4env venv).
# Usage: bash ~/run_koumyou_fdtd.sh   (or: wsl bash ~/run_koumyou_fdtd.sh  from Windows)
set -e
source ~/s4env/bin/activate
cd "/mnt/c/Users/bozu1/OneDrive/デスクトップ/lab/koumyou_sim"
streamlit run legacy/fdtd_app.py

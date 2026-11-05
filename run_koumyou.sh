#!/bin/bash
# Iridis（構造色シミュレータ）の起動スクリプト（WSL / ~/s4env）。
# 計算は S4（RCWA2D/libs4lab.so、未ビルドなら `bash third_party/S4/build_s4lab.sh`）。
# 並列ワーカー数は IRIDIS_WORKERS で変更可（既定: CPU数）。
# Usage: bash run_koumyou.sh   (or: wsl bash ~/run_koumyou.sh  from Windows)
set -e
source ~/s4env/bin/activate
cd "/mnt/c/Users/bozu1/OneDrive/デスクトップ/lab/koumyou_sim"
streamlit run iridis.py

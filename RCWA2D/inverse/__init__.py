"""逆設計: 目標の色（sRGB）または反射スペクトルに近い構造を探す。Streamlit 非依存。

targets.py   目標（色・スペクトル）と、反射率からずれを計算する部分
mie.py       孤立した球の Mie 理論（探索の説明用）
runner.py    S4 を使う構造の探索（粗い計算で広く探す → 細かい計算で寸法を直す → さらに細かく確かめる）
families/    構造の種類ごとの定義
  film.py     平らな多層膜（伝達行列法。差分進化の探索もここ）
  grating.py  1 次元の格子
  columns.py  1 次元の列構造（段差の上に多層膜）
  particles.py  Mie 共鳴の円柱・球の配列（回折が出ない周期に限る）
"""
from .targets import ColorTarget, SpectrumTarget

__all__ = ['ColorTarget', 'SpectrumTarget']

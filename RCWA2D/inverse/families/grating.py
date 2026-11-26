"""1 次元の格子（基板の上に、断面が長方形の棒を周期的に並べたもの）の逆設計。探す寸法は周期・棒の幅・高さ。"""
import numpy as np

from ...structure_builder import columns_to_layers
from ..runner import Level, Var, snap_vars
from .particles import max_period_nm


class Grating:
    name, kind = 'grating', '1d'
    levels = (Level(21, 0, 2, 41), Level(41, 0, 2, 41), Level(81, 0, 1, 81))

    def __init__(self, material, pitch=(160.0, 400.0), height=(40.0, 300.0), w_min=40.0, gap_min=40.0, step=1.0,
                 no_diffraction=None):
        """no_diffraction: None（回折を許す）か、max_period_nm に渡す dict(angles_deg, n_env, nk_sub_fn, side)。"""
        hi = float(pitch[1])
        if no_diffraction is not None:
            hi = min(hi, max_period_nm(lattice='1d', **no_diffraction))
        lo = max(float(pitch[0]), w_min + gap_min)
        if np.floor(hi / step + 1e-9) * step < np.ceil(lo / step - 1e-9) * step:
            raise ValueError(f'周期の範囲がありません（下限 {lo:g} nm、上限 {hi:.0f} nm）。')
        self.material, self.materials, self.label = material, (material,), f'{material} の格子'
        self.w_min, self.gap_min, self.step = float(w_min), float(gap_min), float(step)
        self.variables = [Var('P', '周期', lo, hi, step), Var('W', '棒の幅', w_min, hi - gap_min, step),
                          Var('H', '高さ', float(height[0]), float(height[1]), step)]

    def project(self, X):
        X = np.atleast_2d(np.asarray(X, float)).copy()
        X[:, 1] = np.clip(X[:, 1], self.w_min, X[:, 0] - self.gap_min)
        X = snap_vars(X, self.variables)
        w_hi = np.floor((X[:, 0] - self.gap_min) / self.step + 1e-9) * self.step
        X[:, 1] = np.round(np.minimum(X[:, 1], w_hi), 6)
        return X

    def _columns(self, x):
        P, W, H = (float(v) for v in x)
        return [{'width': W, 'subst_raise': 0.0, 'raise_mode': 'sharp',
                 'film_layers': [{'type': 'simple', 'material': self.material, 'thickness': H}]},
                {'width': round(P - W, 6), 'subst_raise': 0.0, 'raise_mode': 'sharp', 'film_layers': []}]

    def geometry(self, x, level):
        return {'pitch_nm': float(x[0]), 'layers': columns_to_layers(self._columns(x))}

    def describe(self, x):
        P, W, H = (float(v) for v in x)
        return {'summary': f'{self.material} の棒: 周期 {P:g}・幅 {W:g}・高さ {H:g} nm',
                'dims': [['周期', P], ['棒の幅', W], ['すき間', round(P - W, 6)], ['高さ', H]],
                'apply': {'mode': '1d', 'columns': self._columns(x)}}

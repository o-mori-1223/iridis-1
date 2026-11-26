"""逆設計の目標（色・反射スペクトル）と、反射率からずれを計算する部分。構造の種類によらない。"""
from dataclasses import dataclass, field

import numpy as np

from .. import color_metrics as cm

# 画面の青・緑・赤のボタンが入れる色
PRESET_COLORS = {'青': '#1f5fd0', '緑': '#1fa045', '赤': '#d0282e'}


@dataclass
class ColorTarget:
    """目標の色。angles_deg と hexes は同じ長さ（入射角ごとの sRGB）。"""
    angles_deg: list
    hexes: list
    ignore_lightness: bool = False   # True なら明度を比べず、色相と彩度だけ合わせる
    aggregate: str = 'mean'          # 角度ごとの ΔE の 'mean'（平均）か 'max'（最悪の角度）
    kind: str = field(default='color', init=False)

    @property
    def angles(self):
        return [float(a) for a in self.angles_deg]


@dataclass
class SpectrumTarget:
    """目標の反射スペクトル（反射率は 0〜1）。"""
    wl_nm: np.ndarray
    R: np.ndarray
    angle_deg: float = 0.0
    kind: str = field(default='spectrum', init=False)

    @property
    def angles(self):
        return [float(self.angle_deg)]

    def grid(self, n_max):
        """目標の波長（多すぎれば n_max 点に間引いた等間隔の波長）と反射率。"""
        wl, R = np.asarray(self.wl_nm, float), np.asarray(self.R, float)
        order = np.argsort(wl)
        wl, R = wl[order], R[order]
        if wl.size > n_max:
            g = np.linspace(wl[0], wl[-1], n_max)
            R, wl = np.interp(g, wl, R), g
        return wl, R


def wavelengths(target, cie_stride, n_max):
    """計算する波長。色なら 380〜780 nm を 5·cie_stride nm 刻み、スペクトルなら目標の波長（最大 n_max 点）。"""
    return cm.CIE_WL[::cie_stride] if target.kind == 'color' else target.grid(n_max)[0]


def _on_cie(wl, R):
    return R if np.array_equal(wl, cm.CIE_WL) else cm.to_cie_grid(wl, R)


def losses(target, wl, Rs):
    """角度ごとの反射率 Rs（各 [候補, 波長 wl]）→（ずれ [候補], 角度ごとのずれ [候補, 角度]）。
    色は ΔE₀₀、スペクトルは RMSE [%]。"""
    if target.kind == 'color':
        per = np.stack([cm.delta_e2000(cm.spectrum_to_lab(_on_cie(wl, R)), cm.hex_to_lab(h),
                                       ignore_lightness=target.ignore_lightness)
                        for R, h in zip(Rs, target.hexes)], axis=1)
        return (per.max(axis=1) if target.aggregate == 'max' else per.mean(axis=1)), per
    order = np.argsort(target.wl_nm)
    R_t = np.interp(wl, np.asarray(target.wl_nm, float)[order], np.asarray(target.R, float)[order])
    per = 100.0 * np.sqrt(np.mean((Rs[0] - R_t[None, :]) ** 2, axis=1))[:, None]
    return per[:, 0], per


def residuals(target, wl, R):
    """候補 1 つの反射率（角度ごとに [波長]）→ 目標との差のベクトル（色は角度ごとの Lab、スペクトルは反射率 [%]）。
    仕上げの探索（runner._polish）が、寸法を動かしたときの変化の向きを見るのに使う。"""
    if target.kind == 'color':
        out = []
        for r, h in zip(R, target.hexes):
            lab, t = cm.spectrum_to_lab(_on_cie(wl, np.asarray(r)[None, :]))[0], cm.hex_to_lab(h)
            d = (lab - t) / np.array([1.0, 1 + 0.045 * np.hypot(t[1], t[2]), 1 + 0.045 * np.hypot(t[1], t[2])])
            out.append(d[1:] if target.ignore_lightness else d)
        return np.concatenate(out)
    order = np.argsort(target.wl_nm)
    R_t = np.interp(wl, np.asarray(target.wl_nm, float)[order], np.asarray(target.R, float)[order])
    return 100.0 * (np.asarray(R[0]) - R_t) / np.sqrt(len(wl))


def hexes_of(wl, Rs_one):
    """角度ごとの反射率（各 [波長]）→ sRGB の色。"""
    return [cm.xyz_to_hex(cm.spectrum_to_xyz(cm.to_cie_grid(wl, R))) for R in Rs_one]

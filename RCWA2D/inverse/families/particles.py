"""Mie 共鳴の粒子（誘電体の円柱・球を基板の上に 1 段、正方・六方格子に並べたもの）の逆設計。

Mie 共鳴（粒子 1 個の中で起きる共鳴）の色に限るため、回折が出ない周期だけを探す。確かめた候補については、
粒子の寸法を変えずに周期だけを ±5% 動かし、反射のピークがどれだけ動くかも計算する（Mie 共鳴ならほとんど
動かず、格子の影響が大きいと周期と同じ割合で動く）。
"""
import numpy as np

from ...rcwa_s4_array2d_backend import ENV_MATERIAL, sphere_disk_layers
from .. import mie
from ..runner import Level, Var, snap_vars

PERIOD_SHIFT = 0.05
_HEX = 2.0 / np.sqrt(3.0)


def max_period_nm(angles_deg, n_env, nk_sub_fn, lattice='square', side='both', wl_min_nm=380.0, wl_max_nm=780.0):
    """回折が出ない周期の上限 [nm]（これ未満なら、波長 wl_min〜wl_max・全部の入射角で 0 次だけ）。

    条件は 周期 < λ /（出る側の n ＋ 環境の n·sinθ）。六方格子は、格子の列の間隔が格子定数の √3/2 倍なので 2/√3 倍。
    side: 'both' = 空気（環境）側にも基板側にも出ない。'env' = 環境側だけ（基板側に出る回折は許す）。"""
    s = max(np.sin(np.radians(a)) for a in angles_deg) * float(np.real(n_env))
    lim = np.inf
    for w in np.arange(wl_min_nm, wl_max_nm + 0.5, 5.0):
        n_out = float(np.real(n_env))
        if side == 'both':
            n_out = max(n_out, float(np.real(complex(nk_sub_fn(w)))))
        lim = min(lim, w / (n_out + s))
    return lim * (_HEX if lattice == 'hex' else 1.0)


def peak_nm(wl, R, near=None):
    """反射のピークの波長（放物線で内挿）。near を与えると、それに一番近い山。山が無ければ None。"""
    wl, R = np.asarray(wl, float), np.asarray(R, float)
    i = [k for k in range(1, len(R) - 1) if R[k] > R[k - 1] and R[k] >= R[k + 1]]
    if not i:
        return None
    k = max(i, key=lambda j: R[j]) if near is None else min(i, key=lambda j: abs(wl[j] - near))
    a, b, c = R[k - 1], R[k], R[k + 1]
    den = a - 2 * b + c
    return float(wl[k] + (0.5 * (a - c) / den if den != 0 else 0.0) * (wl[k + 1] - wl[k]))


class Particles:
    kind = '2d'

    def __init__(self, material, shape='disk', lattice='square', pitch=(160.0, 400.0), diameter=(60.0, 300.0),
                 height=(40.0, 300.0), gap_min=20.0, aspect_max=2.0, step=1.0, diffraction=None):
        """diffraction: max_period_nm に渡す dict(angles_deg, n_env, nk_sub_fn, side, wl_min_nm)（周期の上限を決める）。"""
        self.name = 'mie_' + shape
        self.material, self.materials, self.shape, self.lattice = material, (material,), shape, lattice
        self.gap_min, self.aspect_max, self.step = float(gap_min), float(aspect_max), float(step)
        self.p_limit = max_period_nm(lattice=lattice, **diffraction) if diffraction else np.inf
        self.n_env = float(np.real(diffraction['n_env'])) if diffraction else 1.0
        hi = min(float(pitch[1]), self.p_limit - 1e-6)
        lo = max(float(pitch[0]), float(diameter[0]) + gap_min)
        if np.floor(hi / step + 1e-9) * step < np.ceil(lo / step - 1e-9) * step:
            raise ValueError(f'周期の範囲がありません（下限 {lo:g} nm、上限 {hi:.0f} nm）。')
        self.label = f'{material} の' + ('円柱' if shape == 'disk' else '球')
        self.variables = [Var('P', '周期', lo, hi, step), Var('D', '直径', float(diameter[0]), min(float(diameter[1]), hi - gap_min), step)]
        if shape == 'disk':
            self.variables.append(Var('H', '高さ', float(height[0]), float(height[1]), step))
            self.levels = (Level(49), Level(121), Level(201, cie_stride=2, n_wl_max=41))
        else:
            self.levels = (Level(49, n_slices=8), Level(81, n_slices=10), Level(121, n_slices=16, cie_stride=2, n_wl_max=41))

    def project(self, X):
        X = np.atleast_2d(np.asarray(X, float)).copy()
        X[:, 1] = np.minimum(X[:, 1], X[:, 0] - self.gap_min)
        X = snap_vars(X, self.variables)
        fl = lambda v: np.floor(v / self.step + 1e-9) * self.step
        X[:, 1] = np.round(np.minimum(X[:, 1], fl(X[:, 0] - self.gap_min)), 6)
        if self.shape == 'disk':
            X[:, 2] = np.round(np.maximum(np.minimum(X[:, 2], fl(self.aspect_max * X[:, 1])), self.variables[2].lo), 6)
        return X

    def _geometry(self, P, D, H, level):
        if self.shape == 'disk':
            layers = [{'thickness_nm': H, 'bg_material': ENV_MATERIAL,
                       'disks': [{'material': self.material, 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': D / 2}]}]
        else:
            layers = sphere_disk_layers(D / 2, self.material, level.n_slices)
        return {'pitch_nm': P, 'gamma_deg': 60.0 if self.lattice == 'hex' else 90.0, 'layers': layers}

    def _values(self, x):
        P, D = float(x[0]), float(x[1])
        return P, D, (float(x[2]) if self.shape == 'disk' else D)

    def geometry(self, x, level):
        return self._geometry(*self._values(x), level)

    def describe(self, x):
        P, D, H = self._values(x)
        dims = [['周期', P], ['直径', D], ['すき間', round(P - D, 6)]] + ([['高さ', H]] if self.shape == 'disk' else [])
        lyr = {'material': self.material, 'bg_material': ENV_MATERIAL, 'radius_nm': D / 2, 'cx_nm': 0.0, 'cy_nm': 0.0,
               'n_slices': 20}
        lyr.update({'type': 'disk', 'thickness_nm': H} if self.shape == 'disk' else {'type': 'sphere'})
        return {'summary': f'{self.label}: 周期 {P:g}・直径 {D:g}' + (f'・高さ {H:g}' if self.shape == 'disk' else '') + ' nm'
                           + ('（六方）' if self.lattice == 'hex' else '（正方）'),
                'dims': dims, 'apply': {'mode': '2d', 'lattice': self.lattice, 'pitch_nm': P, 'array_layers': [lyr]}}

    def post(self, ctx, x, level):
        """周期を ±5% 動かしたときのピークの動きと、同じ体積の孤立した球の Mie 共鳴の波長。"""
        P, D, H = self._values(x)
        wl = np.arange(380.0, 781.0, 10.0)
        periods = [P] + [p for p in (P * (1 - PERIOD_SHIFT), P * (1 + PERIOD_SHIFT)) if D < p < self.p_limit]
        Rs = ctx.solve([self._geometry(p, D, H, level) for p in periods], level, wl, [ctx.target.angles[0]])
        out = {}
        if Rs and all(R is not None for R in Rs):
            p0 = peak_nm(wl, Rs[0][0])
            moved = [(p / P - 1, peak_nm(wl, R[0], near=p0) if p0 else None) for p, R in zip(periods[1:], Rs[1:])]
            slope = [(q / p0 - 1) / dp for dp, q in moved if q]
            out['period_check'] = {'peak_nm': p0, 'moved': [[round(100 * dp, 1), q] for dp, q in moved],
                                   'ratio': float(np.mean(np.abs(slope))) if slope else None}
        v = np.pi * D ** 2 / 4 * H if self.shape == 'disk' else np.pi * D ** 3 / 6
        d_eq = (6 * v / np.pi) ** (1 / 3)
        out['mie_sphere'] = dict(diameter_nm=float(d_eq), **mie.dipole_resonances(ctx.nk_fns[self.material], d_eq, self.n_env))
        return out

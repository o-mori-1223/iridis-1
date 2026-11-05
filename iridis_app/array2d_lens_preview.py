"""レンズ状の凹凸＋形に沿った多層膜の3Dプレビュー用メッシュ（RCWA2D.curved_multilayer の info から描く）。
縦断面図は array2d_cross_section（全構造共通）が描く。"""
import numpy as np
import plotly.graph_objects as go

from .array2d_preview import MESH_LIGHTING

_SUBST_COLOR = "#a0b8cc"


def lens_surface_traces(info, lattice_pts, colors, cx_nm=0.0, cy_nm=0.0, n_phi=36, n_r=40, film_r_max=None):
    """3Dプレビュー用: 各境界面の凹凸部分（影響半径の内側）を回転面のメッシュにして返す。
    平坦部は呼び出し側で直方体として描く。colors は材質 → 色。film_r_max を与えると膜の上面
    （基板表面以外）をその半径までで切る（多層膜円盤）。"""
    x, Z, mats = info['x'], info['Z'], info['materials']
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    traces = []
    names = [('基板表面', _SUBST_COLOR)] + [(f'{k}層目 {m} の上面', colors.get(m, "#7f8c8d"))
                                          for k, m in enumerate(mats, start=1)]
    for kk, (zk, (label, clr)) in enumerate(zip(Z, names)):
        r_max = x[-1] if (kk == 0 or film_r_max is None) else min(float(film_r_max), x[-1])
        r_s = np.linspace(0.0, r_max, n_r)
        z_s = np.interp(r_s, x, zk)
        rr, pp = np.meshgrid(r_s, ph, indexing='ij')
        zz = np.repeat(z_s[:, None], n_phi, axis=1)
        i, j, k = [], [], []
        for a in range(n_r - 1):
            for b in range(n_phi):
                p00, p01 = a * n_phi + b, a * n_phi + (b + 1) % n_phi
                p10, p11 = (a + 1) * n_phi + b, (a + 1) * n_phi + (b + 1) % n_phi
                i += [p00, p00]; j += [p10, p11]; k += [p11, p01]
        for pt in lattice_pts:
            traces.append(go.Mesh3d(
                x=(pt[0] + cx_nm + rr * np.cos(pp)).ravel(), y=(pt[1] + cy_nm + rr * np.sin(pp)).ravel(),
                z=zz.ravel(), i=i, j=j, k=k, color=clr, opacity=0.55, flatshading=True, lighting=MESH_LIGHTING,
                hoverinfo='text', hovertext=label, showscale=False))
    return traces

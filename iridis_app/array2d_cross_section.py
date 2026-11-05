"""2Dモードの縦断面図（S4計算は行わない）。S4 に実際に渡す layers（top-first、各層は背景＋
円/矩形/多角形）を、x 方向（y = 一定）または y 方向（x = 一定）の直線に沿って切った断面として描く。層構成・基板段差・
レンズ状の凹凸のどれでも同じ図になる（球・半球・凹凸は階段近似そのものが見える）。
レンズ状の凹凸では、実際の境界面（なめらかな曲線）を重ねて近似の粗さを確かめられる。"""
import numpy as np
import plotly.graph_objects as go

from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL

from .array2d_model import PREVIEW_PALETTE

_SUBST_COLOR = "#a0b8cc"
_ENV_COLOR = "#ffffff"
_AXIS_COLOR = "#1f2933"
_N_SAMPLES_PER_PERIOD = 1200


def _shapes(L):
    """層の図形 → [(種類, 図形, 面積)]。S4 では入れ子の図形は内側（小さい方）が優先される。"""
    out = []
    for d in L.get('disks', []):
        if d['radius_nm'] > 0:
            out.append(('disk', d, np.pi * d['radius_nm'] ** 2))
    for r in L.get('rects', []):
        if r['hw_x_nm'] > 0 and r['hw_y_nm'] > 0:
            out.append(('rect', r, 4 * r['hw_x_nm'] * r['hw_y_nm']))
    for pg in L.get('polygons', []):
        v = np.asarray(pg['vertices_nm'], float)
        area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
        out.append(('poly', pg, area))
    return out


def _inside(kind, s, px, py):
    """点列 (px, py) が図形の内側か（px, py は同じ形の配列）。"""
    if kind == 'disk':
        return (px - s['cx_nm']) ** 2 + (py - s['cy_nm']) ** 2 < s['radius_nm'] ** 2
    if kind == 'rect':
        return (np.abs(px - s['cx_nm']) < s['hw_x_nm']) & (np.abs(py - s['cy_nm']) < s['hw_y_nm'])
    v = np.asarray(s['vertices_nm'], float)        # 多角形: 偶奇判定（水平方向の半直線）
    inside = np.zeros(px.shape, bool)
    for (x1, y1), (x2, y2) in zip(v, np.roll(v, -1, axis=0)):
        if y1 == y2:
            continue
        cross = ((y1 > py) != (y2 > py)) & (px < x1 + (py - y1) * (x2 - x1) / (y2 - y1))
        inside ^= cross
    return inside


def _bbox(kind, s):
    if kind == 'disk':
        r = s['radius_nm']
        return s['cx_nm'] - r, s['cx_nm'] + r, s['cy_nm'] - r, s['cy_nm'] + r
    if kind == 'rect':
        return s['cx_nm'] - s['hw_x_nm'], s['cx_nm'] + s['hw_x_nm'], s['cy_nm'] - s['hw_y_nm'], s['cy_nm'] + s['hw_y_nm']
    v = np.asarray(s['vertices_nm'], float)
    return v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()


def _materials_at(L, px, py, shifts):
    """層 L の、点列 (px, py)（断面の線上の点）での材質（周期像も含めて判定）。"""
    mats = np.full(len(px), L['bg_material'], dtype=object)
    best = np.full(len(px), np.inf)
    lo_x, hi_x, lo_y, hi_y = px.min(), px.max(), py.min(), py.max()
    for kind, s, area in _shapes(L):
        x0, x1, y0, y1 = _bbox(kind, s)
        hit = np.zeros(len(px), bool)
        for sx, sy in shifts:                       # 図形を格子ベクトルでずらす代わりに点をずらす
            if x1 + sx < lo_x or x0 + sx > hi_x or y1 + sy < lo_y or y0 + sy > hi_y:
                continue                            # この周期像は断面の線にかからない
            sel = (px >= x0 + sx) & (px <= x1 + sx) & (py >= y0 + sy) & (py <= y1 + sy)
            if sel.any():
                hit[sel] |= _inside(kind, s, px[sel] - sx, py[sel] - sy)
        win = hit & (area <= best)                  # 小さい図形（内側）が優先。同じ大きさなら後の図形
        mats[win] = s['material']
        best[win] = area
    return mats


def _materials_along_line(L, xs, y0, shifts):
    """層 L を y = y0 で切ったときの各 x の材質。"""
    return _materials_at(L, xs, np.full(len(xs), float(y0)), shifts)


def stack_cross_section_fig(layers, a1, a2, pos_nm, color_map, subst_label, n_periods=1,
                            z_bottom_nm=0.0, lens_info=None, lens_center=(0.0, 0.0), direction='x'):
    """縦断面。direction='x' なら y = pos で x 方向に、'y' なら x = pos で y 方向に切る。
    layers は top-first（S4 に渡すものと同じ）。z_bottom_nm は最下層の下面の高さ
    （凹のレンズ状の凹凸では負）。横軸の範囲は、その方向の周期 × n_periods。"""
    p = (float(np.linalg.norm(a1)) if direction == 'x'
         else abs(float(a1[0] * a2[1] - a1[1] * a2[0])) / float(np.linalg.norm(a1)))   # a1 に垂直な方向の周期
    half = p * n_periods / 2.0
    xs = np.linspace(-half, half, _N_SAMPLES_PER_PERIOD * n_periods + 1)   # 断面の横軸（x または y）
    px, py = (xs, np.full(len(xs), float(pos_nm))) if direction == 'x' else (np.full(len(xs), float(pos_nm)), xs)
    n = n_periods + 3
    shifts = [tuple(i * a1 + j * a2) for i in range(-n, n + 1) for j in range(-n, n + 1)]

    total = sum(L['thickness_nm'] for L in layers)
    z_top = z_bottom_nm + total
    subst_h = max(0.15 * total, 30.0)
    env_h = max(0.08 * total, 20.0)

    color_map = dict(color_map)

    def color(m):
        if m == SUBST_MATERIAL:
            return _SUBST_COLOR
        if m == ENV_MATERIAL:
            return _ENV_COLOR
        if m not in color_map:                       # 背景にだけ使われる材質など
            color_map[m] = PREVIEW_PALETTE[len(color_map) % len(PREVIEW_PALETTE)]
        return color_map[m]

    def label(m):
        if m == SUBST_MATERIAL:
            return f"基板（{subst_label}）"
        if m == ENV_MATERIAL:
            return "環境"
        return m

    # 材質ごとに長方形をまとめて1トレースにする（None 区切りの塗りつぶし）
    rects = {}
    def add_rect(m, x0, x1, z0, z1):
        px, pz = rects.setdefault(m, ([], []))
        px += [x0, x1, x1, x0, x0, None]
        pz += [z0, z0, z1, z1, z0, None]

    add_rect(SUBST_MATERIAL, -half, half, z_bottom_nm - subst_h, z_bottom_nm)
    add_rect(ENV_MATERIAL, -half, half, z_top, z_top + env_h)
    z = z_top
    bounds_x, bounds_z = [], []
    for L in layers:                                 # 上から
        z_lo = z - L['thickness_nm']
        mats = _materials_at(L, px, py, shifts)
        change = np.nonzero(mats[1:] != mats[:-1])[0]
        starts = np.concatenate([[0], change + 1])
        ends = np.concatenate([change + 1, [len(xs) - 1]])
        for s, e in zip(starts, ends):
            add_rect(mats[s], xs[s], xs[e], z_lo, z)
        bounds_x += [-half, half, None]
        bounds_z += [z_lo, z_lo, None]
        z = z_lo

    fig = go.Figure()
    order = [SUBST_MATERIAL, ENV_MATERIAL] + [m for m in rects if m not in (SUBST_MATERIAL, ENV_MATERIAL)]
    for m in order:
        if m not in rects:
            continue
        rx, rz = rects[m]
        fig.add_trace(go.Scatter(x=rx, y=rz, fill='toself', fillcolor=color(m), mode='lines',
                                 line=dict(width=0, color=color(m)), name=label(m), hoverinfo='name',
                                 showlegend=m != ENV_MATERIAL))
    fig.add_trace(go.Scatter(x=bounds_x, y=bounds_z, mode='lines', line=dict(color='rgba(0,0,0,0.45)', width=0.8),
                             name='S4 の層の境目', hoverinfo='skip'))

    if lens_info is not None:   # 実際の境界面（なめらかな曲線）を重ねる
        # 境界面の高さは「最も近い凹凸の中心からの距離 d」で決まる（curved_multilayer の 4.）。
        x, Z = lens_info['x'], lens_info['Z']
        rd = lens_info.get('disk_radius_nm')
        centers = np.array([(lens_center[0] + sx, lens_center[1] + sy) for sx, sy in shifts])
        d = np.hypot(px[:, None] - centers[None, :, 0], py[:, None] - centers[None, :, 1]).min(axis=1)
        lx, lz = [], []
        for k, zk in enumerate(Z):
            zz = np.interp(d, x, zk)
            if k > 0 and rd is not None:              # 多層膜円盤: 円盤の外に膜はない
                zz = np.where(d <= rd, zz, np.nan)
            lx += list(xs) + [None]
            lz += [None if np.isnan(v) else float(v) for v in zz] + [None]
        fig.add_trace(go.Scatter(x=lx, y=lz, mode='lines', line=dict(color='black', width=1.2, dash='dot'),
                                 name='実際の境界面（なめらかな形）', hoverinfo='skip'))

    axis = dict(title_font=dict(size=14, color=_AXIS_COLOR), tickfont=dict(size=12, color=_AXIS_COLOR),
                showline=True, linecolor=_AXIS_COLOR, ticks='outside', tickcolor=_AXIS_COLOR,
                gridcolor="#d5dbe1", zeroline=False, mirror=True)
    fig.update_layout(
        height=480, margin=dict(l=60, r=10, t=40, b=40), plot_bgcolor='white', paper_bgcolor='white',
        font=dict(color=_AXIS_COLOR),
        title=dict(text=f"縦断面（{'y' if direction == 'x' else 'x'} = {pos_nm:.0f} nm で {direction} 方向に切る、"
                        f"{n_periods}周期）・S4 に渡す {len(layers)} 層", font=dict(size=13)),
        xaxis=dict(axis, title=f'{direction} [nm]', range=[-half, half]),
        yaxis=dict(axis, title='高さ z [nm]', scaleanchor='x', scaleratio=1,
                   range=[z_bottom_nm - subst_h, z_top + env_h]),
        legend=dict(orientation='h', y=-0.18),
    )
    return fig

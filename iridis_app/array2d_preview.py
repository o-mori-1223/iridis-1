"""2Dモードの3D構造プレビュー（S4計算は一切行わない）。"""
import numpy as np
import plotly.graph_objects as go

from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL, _expand_to_bands

from .array2d_model import (PARTICLE_TYPES, expand_layer_spec, hcylinder_footprint_nm, layer_thickness_nm,
                            preview_color_map)

_SUBST_COLOR = "#a0b8cc"
# plotly 既定の照明は縁が白く光る（fresnel・鏡面反射）ため、輪郭がぼやけて見える。拡散光中心にする。
MESH_LIGHTING = dict(ambient=0.55, diffuse=0.75, specular=0.05, roughness=0.9, fresnel=0.0)
_AXIS_COLOR = "#1f2933"


def _mesh(xs, ys, zs, i, j, k, color, opacity, hovertext):
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                     color=color, opacity=opacity, flatshading=True, lighting=MESH_LIGHTING,
                     hoverinfo='text' if hovertext else 'skip', hovertext=hovertext,
                     showscale=False)


def wall_mesh3d(x0, x1, y0, y1, z0, z1, color, opacity=1.0, hovertext=None):
    """直方体の側面4枚だけ（上下面なし）。枠全体を覆う半透明の膜に使う。上下面まで描くと
    半透明の水平面が何十枚も重なり、中の柱や粒子の輪郭がかすんで見えるため。"""
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    tris = [(0,1,5),(0,5,4), (3,2,6),(3,6,7), (0,3,7),(0,7,4), (1,2,6),(1,6,5)]
    i, j, k = zip(*tris)
    return _mesh(xs, ys, zs, i, j, k, color, opacity, hovertext)


class _Edges:
    """輪郭線を1本の Scatter3d にまとめる（トレース数を増やさずに境界をくっきり描く）。"""

    def __init__(self):
        self.xs, self.ys, self.zs = [], [], []

    def polyline(self, xs, ys, zs):
        self.xs += list(xs) + [None]; self.ys += list(ys) + [None]; self.zs += list(zs) + [None]

    def circle(self, cx, cy, r, z, n_seg):
        t = np.linspace(0, 2 * np.pi, n_seg + 1)
        self.polyline(cx + r * np.cos(t), cy + r * np.sin(t), np.full(n_seg + 1, z))

    def rect(self, x0, x1, y0, y1, z):
        self.polyline([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5)

    def trace(self, color='rgba(20,30,40,0.55)', width=1.5):
        return go.Scatter3d(x=self.xs, y=self.ys, z=self.zs, mode='lines', line=dict(color=color, width=width),
                            hoverinfo='skip', showlegend=False)


def _axis(title):
    return dict(title=dict(text=title, font=dict(size=15, color=_AXIS_COLOR)),
                tickfont=dict(size=13, color=_AXIS_COLOR), nticks=6,
                showbackground=True, backgroundcolor="#f3f5f7",
                gridcolor="#b8c2cc", gridwidth=1.5, zerolinecolor="#5b6772", zerolinewidth=2,
                showline=True, linecolor=_AXIS_COLOR, linewidth=2,
                ticks='outside', tickcolor=_AXIS_COLOR, ticklen=6)


def box_mesh3d(x0, x1, y0, y1, z0, z1, color, opacity=1.0, hovertext=None):
    """軸並行直方体（8頂点・12三角形）。平坦膜・基板・粒子層の背景スラブに使う。"""
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    tris = [(0,1,2),(0,2,3), (4,5,6),(4,6,7),
            (0,1,5),(0,5,4), (3,2,6),(3,6,7),
            (0,3,7),(0,7,4), (1,2,6),(1,6,5)]
    i, j, k = zip(*tris)
    return _mesh(xs, ys, zs, i, j, k, color, opacity, hovertext)


def cylinder_mesh3d(cx, cy, r, z0, z1, color, opacity=1.0, n_seg=64, hovertext=None, caps=True):
    """z軸方向に押し出した円柱。円盤層の粒子・円形の上げ領域に使う。caps=False なら側面だけ。"""
    theta = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
    ring_x, ring_y = cx + r*np.cos(theta), cy + r*np.sin(theta)
    xs = np.concatenate([ring_x, ring_x, [cx, cx]])
    ys = np.concatenate([ring_y, ring_y, [cy, cy]])
    zs = np.concatenate([np.full(n_seg, z0), np.full(n_seg, z1), [z0, z1]])
    bot_c, top_c = 2*n_seg, 2*n_seg + 1
    i, j, k = [], [], []
    for s in range(n_seg):
        s2 = (s + 1) % n_seg
        i += [s, s]; j += [s2, s2 + n_seg]; k += [s2 + n_seg, s + n_seg]
        if caps:
            i += [bot_c, top_c]; j += [s2, s + n_seg]; k += [s, s2 + n_seg]
    return _mesh(xs, ys, zs, i, j, k, color, opacity, hovertext)


def sphere_mesh3d(cx, cy, cz, r, color, opacity=1.0, n_theta=14, n_phi=20, hovertext=None):
    """球（緯度経度グリッド分割）。球層の粒子に使う。"""
    th, ph = np.meshgrid(np.linspace(0, np.pi, n_theta),
                         np.linspace(0, 2*np.pi, n_phi, endpoint=False), indexing='ij')
    xs = (cx + r*np.sin(th)*np.cos(ph)).ravel()
    ys = (cy + r*np.sin(th)*np.sin(ph)).ravel()
    zs = (cz + r*np.cos(th)).ravel()

    def idx(ti, pi): return ti*n_phi + (pi % n_phi)

    i, j, k = [], [], []
    for ti in range(n_theta - 1):
        for pi in range(n_phi):
            a, b = idx(ti, pi), idx(ti, pi+1)
            c, d = idx(ti+1, pi), idx(ti+1, pi+1)
            i += [a, a]; j += [b, d]; k += [d, c]
    return _mesh(xs, ys, zs, i, j, k, color, opacity, hovertext)


def hemisphere_mesh3d(cx, cy, z0, r, color, opacity=1.0, n_theta=8, n_phi=20, hovertext=None):
    """平らな面を下（z=z0）にしたドーム形の半球（緯度経度グリッド＋底面の扇形）。"""
    th, ph = np.meshgrid(np.linspace(0, np.pi / 2, n_theta),
                         np.linspace(0, 2*np.pi, n_phi, endpoint=False), indexing='ij')
    xs = np.append((cx + r*np.sin(th)*np.cos(ph)).ravel(), cx)
    ys = np.append((cy + r*np.sin(th)*np.sin(ph)).ravel(), cy)
    zs = np.append((z0 + r*np.cos(th)).ravel(), z0)
    base_c = n_theta * n_phi

    def idx(ti, pi): return ti*n_phi + (pi % n_phi)

    i, j, k = [], [], []
    for ti in range(n_theta - 1):
        for pi in range(n_phi):
            a, b = idx(ti, pi), idx(ti, pi+1)
            c, d = idx(ti+1, pi), idx(ti+1, pi+1)
            i += [a, a]; j += [b, d]; k += [d, c]
    for pi in range(n_phi):   # 底面（赤道リングと中心を結ぶ扇形）
        i.append(base_c); j.append(idx(n_theta - 1, pi)); k.append(idx(n_theta - 1, pi + 1))
    return _mesh(xs, ys, zs, i, j, k, color, opacity, hovertext)


def hcylinder_mesh3d(cx, cy, zc, r, length, axis, color, opacity=1.0, n_seg=48, hovertext=None):
    """横に倒したシリンダー（軸が x/y 方向、中心軸の高さ zc、端面は平ら）。"""
    t = np.linspace(0, 2*np.pi, n_seg, endpoint=False)
    u = np.concatenate([np.full(n_seg, -length / 2), np.full(n_seg, length / 2), [-length / 2, length / 2]])
    v = np.concatenate([r*np.cos(t), r*np.cos(t), [0.0, 0.0]])      # 軸に垂直な水平方向
    zs = np.concatenate([zc + r*np.sin(t), zc + r*np.sin(t), [zc, zc]])
    xs, ys = (cx + u, cy + v) if axis == 'x' else (cx + v, cy + u)
    end0, end1 = 2*n_seg, 2*n_seg + 1
    i, j, k = [], [], []
    for s in range(n_seg):
        s2 = (s + 1) % n_seg
        i += [s, s, end0, end1]
        j += [s2, s2 + n_seg, s2, s + n_seg]
        k += [s2 + n_seg, s + n_seg, s, s2 + n_seg]
    return _mesh(xs, ys, zs, i, j, k, color, opacity, hovertext)


def _unit_cell_outline(a1, a2, z0, z1):
    """原点の格子点を中心とする単位胞（a1, a2 の平行四辺形）の枠を、上下面＋縦の辺の線で描く。"""
    corners = [-(a1 + a2) / 2, (a1 - a2) / 2, (a1 + a2) / 2, (a2 - a1) / 2]
    xs, ys, zs = [], [], []
    for z in (z0, z1):
        for c in corners + corners[:1]:
            xs.append(c[0]); ys.append(c[1]); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    for c in corners:
        xs += [c[0], c[0], None]; ys += [c[1], c[1], None]; zs += [z0, z1, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='black', width=4),
                        hoverinfo='skip', showlegend=False)


def structure_preview_fig(array_layers, a1, a2, subst_label, lattice_shape,
                          step_regions=None, step_film=(), n_side=0, lens_info=None, lens_center=(0.0, 0.0),
                          supercell=None):
    """多層スタック（平坦膜/円盤/球）＋基板段差の3Dプレビュー。表示する格子点は
    n_side=0 なら中央の1周期だけ、n_side=1 なら中心周囲3×3（全周期を描くと粒子層が多い
    構造でメッシュ数が膨大になるため。S4へ渡すのは常に1周期分で、計算結果には影響しない）。
    表示枠は表示している単位胞の外周にぴったり合わせる（寸法が周期と比べられるように）。
    step_regions を与えた場合、基板段差（コンフォーマル多層）を基板のすぐ上に描く
    （_expand_to_bands で field=上げ0・各領域=raise_nm 上げのバンドに展開。S4用の
    regions_to_layers_2d と同じ規約）。
    supercell（周期領域と円柱の切り抜きの設定、array2d_supercell_editor）を与えると、周期領域の Nx×Ny 周期を
    描き、黒枠を周期領域にし、切り抜く円柱を赤の半透明の壁で重ねる（形そのものは切り抜く前のまま）。"""
    layers_list = [sub for lyr in array_layers for sub in expand_layer_spec(lyr)]
    color_map = preview_color_map(array_layers, step_film)
    step_regions = step_regions or []
    field_bands = _expand_to_bands({'subst_raise': 0.0, 'film_layers': list(step_film)}) if step_regions else []
    region_bands_list = [_expand_to_bands({'subst_raise': r.get('raise_nm', 0.0), 'film_layers': list(step_film)})
                         for r in step_regions]
    step_total_h = max([field_bands[-1][1] if field_bands else 0.0] +
                       [rb[-1][1] if rb else 0.0 for rb in region_bands_list])
    lens_rd = lens_info.get('disk_radius_nm') if lens_info is not None else None
    if lens_info is not None:   # レンズ状の凹凸: 平坦部の膜を field_bands として描き、凹凸部は回転面で描く
        Zl = lens_info['Z']
        if lens_rd is None:   # 多層膜円盤なら円盤の外に膜はない
            field_bands = [(float(Zl[k - 1][-1]), float(Zl[k][-1]), m)
                           for k, m in enumerate(lens_info['materials'], start=1)]
        step_total_h = float(Zl[-1].max())

    total_h = sum(layer_thickness_nm(l) for l in layers_list) + step_total_h
    if total_h <= 0.0:
        total_h = 100.0
    subst_h = max(total_h * 0.35, 40.0)

    if supercell:
        from RCWA2D.cylinder_cut import supercell_vectors, tile_offsets
        lattice_pts = tile_offsets(a1, a2, supercell['nx'], supercell['ny'])
        gap = supercell.get('gap_nm', 0.0) if supercell.get('cut') and supercell.get('mode') == 'keep' else 0.0
        S1, S2 = supercell_vectors(a1, a2, supercell['nx'], supercell['ny'], gap)
    else:
        rng = range(-n_side, n_side + 1)
        lattice_pts = [i * a1 + j * a2 for i in rng for j in rng]
        S1, S2 = a1, a2
    # 枠 = 表示している単位胞（各格子点を中心とする a1,a2 の平行四辺形）の外接矩形
    corners = [p + s * a1 / 2 + t * a2 / 2 for p in lattice_pts for s in (-1, 1) for t in (-1, 1)]
    fx0, fx1 = min(c[0] for c in corners), max(c[0] for c in corners)
    fy0, fy1 = min(c[1] for c in corners), max(c[1] for c in corners)
    particle_layers = [l for l in layers_list if l['type'] in PARTICLE_TYPES]
    for l in particle_layers:   # 位置オフセットで単位胞からはみ出す粒子も枠に含める
        hx, hy = (hcylinder_footprint_nm(l, a1, a2) if l['type'] == 'hcylinder'
                  else (l['radius_nm'], l['radius_nm']))
        for pt in lattice_pts:
            cx, cy = pt[0] + l.get('cx_nm', 0.0), pt[1] + l.get('cy_nm', 0.0)
            fx0, fx1 = min(fx0, cx - hx), max(fx1, cx + hx)
            fy0, fy1 = min(fy0, cy - hy), max(fy1, cy + hy)
    for r in step_regions:   # 上げ領域も格子点ごとに複製して描くので、その全体を枠に含める
        is_c = r.get('shape') == 'circle'
        rx = r.get('radius_nm', 0.0) if is_c else r.get('hw_x_nm', 0.0)
        ry = r.get('radius_nm', 0.0) if is_c else r.get('hw_y_nm', 0.0)
        for pt in lattice_pts:
            cx, cy = pt[0] + r.get('cx_nm', 0.0), pt[1] + r.get('cy_nm', 0.0)
            fx0, fx1 = min(fx0, cx - rx), max(fx1, cx + rx)
            fy0, fy1 = min(fy0, cy - ry), max(fy1, cy + ry)

    def mat_label(name):
        if name == ENV_MATERIAL: return '環境と同じ'
        if name == SUBST_MATERIAL: return f'基板と同じ ({subst_label})'
        return name

    def mat_color(name, default):
        return _SUBST_COLOR if name == SUBST_MATERIAL else default

    traces = [box_mesh3d(fx0, fx1, fy0, fy1, -subst_h, 0.0, color=_SUBST_COLOR, opacity=1.0,
                         hovertext=f"基板: {subst_label}")]
    edges = _Edges()
    edges.rect(fx0, fx1, fy0, fy1, 0.0)

    # 基板段差: フィールド（上げ領域の外）と各上げ領域を、それぞれのバンド通りに積む。
    # フィールドの膜は枠全体を覆うため、上げ領域の柱が埋もれないよう側面だけを半透明で描き、
    # 境目は線で示す。上げ領域は周期構造なので、表示している各格子点に複製して描く。
    for (h_lo, h_hi, mat, *_r) in field_bands:
        traces.append(wall_mesh3d(fx0, fx1, fy0, fy1, h_lo, h_hi,
                                  color=mat_color(mat, color_map.get(mat, "#7f8c8d")), opacity=0.35,
                                  hovertext=f"{mat_label(mat)}<br>フィールド {h_hi - h_lo:.0f}nm"))
        edges.rect(fx0, fx1, fy0, fy1, h_hi)
    if lens_info is not None:
        from .array2d_lens_preview import lens_surface_traces
        r_clip = lens_info['r_in'] or float(lens_info['x'][-1])
        clipped = dict(lens_info, x=lens_info['x'][lens_info['x'] <= r_clip],
                       Z=lens_info['Z'][:, lens_info['x'] <= r_clip])
        traces += lens_surface_traces(clipped, lattice_pts, color_map, *lens_center, film_r_max=lens_rd)
        if lens_rd is not None:   # 多層膜円盤の側面（垂直）を膜ごとの色の帯で描く
            zr = [float(np.interp(lens_rd, lens_info['x'], zk)) for zk in lens_info['Z']]
            for pt in lattice_pts:
                px_, py_ = pt[0] + lens_center[0], pt[1] + lens_center[1]
                for k, m in enumerate(lens_info['materials'], start=1):
                    if zr[k] - zr[k - 1] > 1e-6:
                        traces.append(cylinder_mesh3d(px_, py_, lens_rd, zr[k - 1], zr[k], color=color_map.get(m, "#7f8c8d"), caps=False,
                                                      hovertext=f"{k}層目 {mat_label(m)}（円盤の側面）"))
                for z in (zr[0], zr[-1]):
                    edges.circle(px_, py_, lens_rd, z, 64)
    for reg, bands in zip(step_regions, region_bands_list):
        for pt in lattice_pts:
            cx, cy = pt[0] + reg.get('cx_nm', 0.0), pt[1] + reg.get('cy_nm', 0.0)
            for (h_lo, h_hi, mat, *_r) in bands:
                clr = mat_color(mat, color_map.get(mat, "#7f8c8d"))
                hover = f"{mat_label(mat)}<br>上げ領域 {h_hi - h_lo:.0f}nm"
                if reg.get('shape') == 'circle':
                    traces.append(cylinder_mesh3d(cx, cy, reg.get('radius_nm', 0.0), h_lo, h_hi,
                                                  color=clr, opacity=1.0, hovertext=hover))
                    edges.circle(cx, cy, reg.get('radius_nm', 0.0), h_hi, 64)
                else:
                    hw_x, hw_y = reg.get('hw_x_nm', 0.0), reg.get('hw_y_nm', 0.0)
                    traces.append(box_mesh3d(cx - hw_x, cx + hw_x, cy - hw_y, cy + hw_y, h_lo, h_hi,
                                             color=clr, opacity=1.0, hovertext=hover))
                    edges.rect(cx - hw_x, cx + hw_x, cy - hw_y, cy + hw_y, h_hi)

    z_cur = step_total_h   # 基板段差の上に「層構成」を積む（top-first格納なので逆順）
    for lyr in reversed(layers_list):
        th = layer_thickness_nm(lyr)
        clr = mat_color(lyr['material'], color_map[lyr['material']])
        if lyr['type'] == 'film':
            traces.append(box_mesh3d(fx0, fx1, fy0, fy1, z_cur, z_cur + th, color=clr, opacity=0.9,
                                     hovertext=f"{mat_label(lyr['material'])}<br>{th:.0f}nm"))
        else:
            r_ = lyr['radius_nm']
            bg = lyr['bg_material']
            traces.append(wall_mesh3d(fx0, fx1, fy0, fy1, z_cur, z_cur + th,
                                      color=mat_color(bg, "#d8dde2" if bg == ENV_MATERIAL else "#c8dcee"),
                                      opacity=0.25, hovertext=f"背景: {mat_label(bg)}"))
            hover = f"{mat_label(lyr['material'])}<br>r={r_:.0f}nm"
            ox, oy = lyr.get('cx_nm', 0.0), lyr.get('cy_nm', 0.0)
            for pt in lattice_pts:
                px_, py_ = pt[0] + ox, pt[1] + oy
                if lyr['type'] == 'sphere':
                    traces.append(sphere_mesh3d(px_, py_, z_cur + r_, r_, color=clr, opacity=0.95, hovertext=hover))
                elif lyr['type'] == 'hemisphere':
                    traces.append(hemisphere_mesh3d(px_, py_, z_cur, r_, color=clr, opacity=0.95, hovertext=hover))
                elif lyr['type'] == 'hcylinder':
                    hx, hy = hcylinder_footprint_nm(lyr, a1, a2)
                    traces.append(hcylinder_mesh3d(px_, py_, z_cur + r_, r_, 2 * (hx if lyr['axis'] == 'x' else hy),
                                                   lyr['axis'], color=clr, opacity=1.0, hovertext=hover))
                else:
                    traces.append(cylinder_mesh3d(px_, py_, r_, z_cur, z_cur + th, color=clr, opacity=1.0,
                                                  hovertext=hover))
                    edges.circle(px_, py_, r_, z_cur, 64)
                    edges.circle(px_, py_, r_, z_cur + th, 64)
        edges.rect(fx0, fx1, fy0, fy1, z_cur + th)
        z_cur += th

    top = max(z_cur, step_total_h)
    cut_text = ""
    if supercell and supercell.get('cut'):   # 切り抜く円柱の壁（半透明）
        R = supercell['radius_nm']
        ccx, ccy = supercell.get('cx_nm', 0.0), supercell.get('cy_nm', 0.0)
        z0 = -float(supercell.get('subst_depth_nm', 0.0))
        traces.append(cylinder_mesh3d(ccx, ccy, R, z0, top, color="#d1495b", opacity=0.22, caps=False,
                                      n_seg=96, hovertext=f"切り抜く円柱 R={R:.0f} nm"))
        for z in (z0, top):
            edges.circle(ccx, ccy, R, z, 96)
        cut_text = (f"・円柱 R={R:.0f} nm で{'切り抜き（外側を取り除く）' if supercell['mode'] == 'keep' else '穴あけ'}"
                    "（赤・半透明）")
    traces.append(edges.trace())
    traces.append(_unit_cell_outline(S1, S2, -subst_h, top))
    fig = go.Figure(data=traces)
    fig.update_layout(
        height=560, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white", font=dict(color=_AXIS_COLOR),
        scene=dict(xaxis=_axis("x [nm]"), yaxis=_axis("y [nm]"), zaxis=_axis("z [nm]"),
                   aspectmode="data", camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))),
        title=dict(text=f"{lattice_shape}（周期 {np.linalg.norm(a1):.0f} nm）・ "
                        + (f"{supercell['nx']}×{supercell['ny']} 周期を表示（黒枠 = 計算する周期領域）{cut_text}"
                           if supercell else
                           "1周期（単位胞）を表示" if n_side == 0 else "中心周囲3×3周期を表示（黒枠 = 計算する単位胞）")
                        + ("・基板段差あり（フィールドの膜は半透明）" if step_regions else "")
                        + ("" if lens_info is None else
                           f"・レンズ状の凹凸＋多層膜円盤 R={lens_rd:.0f} nm（各境界面を半透明で表示）" if lens_rd is not None else
                           "・レンズ状の凹凸あり（平坦部の膜は半透明、凹凸部は各境界面を表示）"),
                   font=dict(size=13)),
    )
    return fig

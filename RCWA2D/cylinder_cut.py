"""2D周期アレイを N×N の周期領域（スーパーセル）に並べ、円柱で切り抜く（S4 に渡す layers の後処理）。

    layers（1周期分） --tile_layers--> layers（Nx×Ny 周期分） --clip_layers_to_cylinder--> layers（切り抜き後）

layers は rcwa_s4_array2d_backend.build_array_geometry と同じ形（top-first、各層は
{'thickness_nm', 'bg_material', 'disks', 'rects', 'polygons'}）。依存は numpy だけ。

■ 並べる（tile_layers）
  各層の円・矩形・多角形を Nx×Ny 個に複製し、(i-(Nx-1)/2)·a1 + (j-(Ny-1)/2)·a2 だけずらす（周期領域の
  中心が原点に来る）。格子ベクトルは Nx·a1、Ny·a2（gap_nm を与えると各方向に gap だけ長くする。
  周期領域どうしの間に余白ができ、余白は切り抜きの外側と同じ材質になる → 切り抜き（内側を残す）専用）。
  spec['arrangement'] が 'square' / 'hex' なら、周期領域の格子ベクトルは中の格子と無関係に、円柱の正方配置 /
  六方最密配置にする（region_lattice。切り抜き・内側を残す専用）。
  単位胞いっぱいの矩形・円は、並べる前に層の背景へ置き換える（simplify_full_cell_regions。並べると
  隣どうしがぴったり接する形になり、S4 の偏光分解が失敗するため）。並べただけなら元の周期構造と厳密に同じ。

■ 切り抜く（clip_layers_to_cylinder）  中心 center・半径 R の円柱
  S4 では同じ層の形どうしは「完全に内側」か「離れている」でないといけない（一部だけ重なるのは不可）。
  円の境界にかかる形はそれを破るので、多角形に直す。円は正多角形 K（頂点は円周上、弦と円弧のずれ ≤ tol_nm）で近似する。
  mode='keep'（内側を残す）:
    背景 = 外側の材質。円 R（材質 = 元の背景、または円を丸ごと含む形の材質）を置き、
    中の形は「円の中 → そのまま」「円の外 → 削除」「境界にかかる → K との交わりの多角形」
    （Sutherland–Hodgman。K は凸なので、元の形が凸でも凹でも K の内側に切り取れる）。
  mode='hole'（内側を抜く）:
    背景はそのまま。穴 = 多角形 K（外側の材質）。中の形は「円の中 → 削除」「円の外 → そのまま」
    「境界にかかる → 形から K を引いた部分」。引いた部分は凹んだ形や 2 つに分かれた形になるので、
    K の各辺の直線で順に切り分けた凸のピース（辺で接するだけで重ならない）の集まりで表す。
    凸でない多角形（レンズ状の凹凸の角の島など）が境界にかかる場合は ValueError（まれ）。
  subst_depth_nm > 0 なら、基板の上側をその厚さだけ切り抜きの対象に含める（基板材質の層を最下部に足す）。
"""
import math

import numpy as np

from .rcwa_s4_array2d_backend import SUBST_MATERIAL, simplify_full_cell_regions

_EPS = 1e-9


# ---------------------------------------------------------------------------
# 周期領域
# ---------------------------------------------------------------------------
def supercell_vectors(a1, a2, nx, ny, gap_nm=0.0):
    """周期領域の格子ベクトル (A1, A2)。gap_nm は各方向に足す余白。"""
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    u1, u2 = a1 / np.linalg.norm(a1), a2 / np.linalg.norm(a2)
    return nx * a1 + gap_nm * u1, ny * a2 + gap_nm * u2


def inscribed_radius(A1, A2):
    """平行四辺形 A1, A2 の内接円の半径（= 2 組の対辺の間隔の短い方の半分）。"""
    area = abs(A1[0] * A2[1] - A1[1] * A2[0])
    return 0.5 * min(area / np.linalg.norm(A1), area / np.linalg.norm(A2))


# 周期領域（切り抜いた円柱）の並べ方。'tile' は単位胞の格子をそのまま Nx×Ny 倍（旧来・spec に無いときの既定）。
# 'square' / 'hex' は円柱を正方配置 / 六方最密配置に並べる（中の格子とは別。円柱の外側は外側の材質で埋まるので、
# 「円柱で切り抜く・内側を残す」ときだけ選べる）。円柱の中心どうしの間隔 D = 2·(Nx×Ny 周期の内接円の半径) + 余白。
ARRANGEMENTS = ('tile', 'square', 'hex')


def arrangement_of(spec):
    """spec で実際に使う並べ方（切り抜かない・穴あけのときは中の格子に合わせるしかないので 'tile'）。"""
    if not spec or not spec.get('cut') or spec.get('mode', 'keep') != 'keep':
        return 'tile'
    arr = spec.get('arrangement', 'tile')
    if arr not in ARRANGEMENTS:
        raise ValueError(f"周期領域の並べ方は {ARRANGEMENTS} のどれか: {arr!r}")
    return arr


def diameter_mode(spec):
    """円柱の直径を直接指定しているか（spec['size_mode'] == 'diameter'。並べ方が正方／六方のときだけ有効）。"""
    return bool(spec) and spec.get('size_mode') == 'diameter' and arrangement_of(spec) != 'tile'


def tiles_for_diameter(a1, a2, diameter_nm, shift_nm=0.0, n_max=61):
    """直径 diameter_nm の円柱（中心のずれ shift_nm）の中を中の格子で埋めるのに要る並べる数 N（奇数、N×N）。
    並べた平行四辺形 (N·a1, N·a2) の内接円が「円柱の半径 + ずれ + 単位胞の対角線」以上になる最小の奇数
    （対角線ぶんの余裕で、隣の単位胞からはみ出してくる粒子も円柱の中に入る）。奇数なので単位胞の中心が原点に来る。"""
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    need = 0.5 * float(diameter_nm) + float(shift_nm) + max(np.linalg.norm(a1 + a2), np.linalg.norm(a1 - a2))
    r1 = inscribed_radius(a1, a2)   # N 倍すると内接円も N 倍
    n = max(1, int(math.ceil(need / r1)))
    n += (n + 1) % 2
    if n > n_max:
        raise ValueError(f"円柱の直径 {diameter_nm:.0f} nm に対して中の格子が細かすぎます（{n}×{n} 個並べる必要がある。"
                         f"上限 {n_max}×{n_max}）。")
    return n


def region_lattice(a1, a2, spec):
    """spec（iridis_app/array2d_supercell_editor.py）どおりの周期領域の格子ベクトル (A1, A2)。A1 は x 軸向き
    （a1 が x 軸向きなら 'tile' も同じ）。spec が None なら (a1, a2)。
    円柱の中心どうしの間隔 D は、直径を指定していれば「直径 + 余白」、そうでなければ「Nx×Ny 周期の内接円の直径 + 余白」。"""
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    if not spec:
        return a1, a2
    nx, ny = int(spec['nx']), int(spec['ny'])
    arr = arrangement_of(spec)
    gap = float(spec.get('gap_nm', 0.0)) if spec.get('cut') and spec.get('mode', 'keep') == 'keep' else 0.0
    if arr == 'tile':
        return supercell_vectors(a1, a2, nx, ny, gap)
    if diameter_mode(spec):
        D = float(spec['diameter_nm']) + gap
    else:
        D = 2.0 * inscribed_radius(*supercell_vectors(a1, a2, nx, ny)) + gap
    if arr == 'square':
        return np.array([D, 0.0]), np.array([0.0, D])
    return np.array([D, 0.0]), np.array([0.5 * D, 0.5 * math.sqrt(3.0) * D])


def cut_radius_limit(a1, a2, spec):
    """切り抜く円柱の「半径 + 中心のずれ」の上限。'tile' は周期領域の内接円（旧来どおり）。'square' / 'hex' は
    並べた Nx×Ny 周期の内接円（それより外には並べた形が無い。間隔 D はこの 2 倍＋余白なので隣とも重ならない）。"""
    if arrangement_of(spec) == 'tile':
        return inscribed_radius(*region_lattice(a1, a2, spec))
    r_tiles = inscribed_radius(*supercell_vectors(a1, a2, int(spec['nx']), int(spec['ny'])))
    if diameter_mode(spec):   # 隣の円柱との間隔 D/2 も上限（直径 ≤ D なので普通はこちらで決まる）
        return min(r_tiles, 0.5 * np.linalg.norm(region_lattice(a1, a2, spec)[0]))
    return r_tiles


def tile_offsets(a1, a2, nx, ny):
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    return [(i - (nx - 1) / 2.0) * a1 + (j - (ny - 1) / 2.0) * a2 for i in range(nx) for j in range(ny)]


def _unit_pitch(a1, a2):
    px, py = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
    gamma = math.degrees(math.atan2(a2[1], a2[0]) - math.atan2(a1[1], a1[0]))
    return px, py, round(gamma, 9)


def tile_layers(layers, a1, a2, nx, ny):
    """1周期分の layers を Nx×Ny 周期分に並べる（元の layers は変えない）。"""
    px, py, gamma = _unit_pitch(a1, a2)
    layers = simplify_full_cell_regions(layers, px / 1000.0, py / 1000.0, gamma)
    if nx == 1 and ny == 1:
        return [dict(L) for L in layers]
    offs = tile_offsets(a1, a2, nx, ny)
    out = []
    for L in layers:
        T = {k: v for k, v in L.items() if k not in ('disks', 'rects', 'polygons')}
        T['disks'] = [dict(d, cx_nm=d['cx_nm'] + o[0], cy_nm=d['cy_nm'] + o[1]) for o in offs for d in L.get('disks', [])]
        T['rects'] = [dict(r, cx_nm=r['cx_nm'] + o[0], cy_nm=r['cy_nm'] + o[1]) for o in offs for r in L.get('rects', [])]
        T['polygons'] = [dict(pg, vertices_nm=[(x + o[0], y + o[1]) for x, y in pg['vertices_nm']])
                         for o in offs for pg in L.get('polygons', [])]
        out.append(T)
    return out


# ---------------------------------------------------------------------------
# 図形
# ---------------------------------------------------------------------------
def circle_polygon(cx, cy, r, tol_nm=0.05, min_n=24):
    """半径 r の円に内接する正多角形（反時計回り）。弦と円弧のずれ（矢高）が tol_nm 以下になる頂点数。"""
    n = min_n if r <= tol_nm else max(min_n, int(math.ceil(math.pi / math.acos(max(-1.0, 1.0 - tol_nm / r)))))
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in t]


def polygon_area(vs):
    if len(vs) < 3:
        return 0.0
    x = np.array([v[0] for v in vs]); y = np.array([v[1] for v in vs])
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ccw(vs):
    return vs if polygon_area(vs) >= 0 else vs[::-1]


def _is_convex(vs):
    n, sign = len(vs), 0
    for i in range(n):
        (x0, y0), (x1, y1), (x2, y2) = vs[i], vs[(i + 1) % n], vs[(i + 2) % n]
        c = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(c) > 1e-12:
            if sign and (c > 0) != (sign > 0):
                return False
            sign = 1 if c > 0 else -1
    return True


def _clip_halfplane(vs, p, q, keep_left=True):
    """多角形 vs を、直線 p→q の左側（keep_left=False なら右側）に切り取る（Sutherland–Hodgman の 1 段）。"""
    def side(v):
        s = (q[0] - p[0]) * (v[1] - p[1]) - (q[1] - p[1]) * (v[0] - p[0])
        return s if keep_left else -s
    out, n = [], len(vs)
    for i in range(n):
        a, b = vs[i], vs[(i + 1) % n]
        sa, sb = side(a), side(b)
        if sa >= 0:
            out.append(a)
        if (sa >= 0) != (sb >= 0) and abs(sa - sb) > 0:
            t = sa / (sa - sb)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _dedupe(vs, tol=1e-9):
    out = []
    for v in vs:
        if not out or abs(v[0] - out[-1][0]) > tol or abs(v[1] - out[-1][1]) > tol:
            out.append(v)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def clip_to_convex(vs, K):
    """多角形 vs ∩ 凸多角形 K（どちらも反時計回り）。"""
    out = list(vs)
    for i in range(len(K)):
        if len(out) < 3:
            return []
        out = _clip_halfplane(out, K[i], K[(i + 1) % len(K)], keep_left=True)
    out = _dedupe(out)
    return out if len(out) >= 3 and polygon_area(out) > 1e-6 else []


def subtract_convex(vs, K):
    """凸多角形 vs から凸多角形 K を引いた部分を、辺で接する凸のピースのリストで返す。"""
    pieces, rest = [], list(vs)
    for i in range(len(K)):
        if len(rest) < 3:
            break
        p, q = K[i], K[(i + 1) % len(K)]
        outside = _dedupe(_clip_halfplane(rest, p, q, keep_left=False))
        if len(outside) >= 3 and polygon_area(outside) > 1e-6:
            pieces.append(outside)
        rest = _dedupe(_clip_halfplane(rest, p, q, keep_left=True))
    return pieces


# ---------------------------------------------------------------------------
# 形と円の関係
# ---------------------------------------------------------------------------
def _shape_polygon(kind, s, tol_nm):
    if kind == 'disk':
        return circle_polygon(s['cx_nm'], s['cy_nm'], s['radius_nm'], tol_nm)
    if kind == 'rect':
        cx, cy, hx, hy = s['cx_nm'], s['cy_nm'], s['hw_x_nm'], s['hw_y_nm']
        return [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
    return _ccw([tuple(v) for v in s['vertices_nm']])


def _point_in_polygon(x, y, vs):
    inside, n = False, len(vs)
    for i in range(n):
        (x0, y0), (x1, y1) = vs[i], vs[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _seg_dist(px, py, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / L2))
    return math.hypot(px - a[0] - t * dx, py - a[1] - t * dy)


def relation(kind, s, cx, cy, R):
    """形 s と円（中心 (cx,cy)、半径 R）の関係: 'in'（円の中）/ 'out'（円の外）/ 'contains'（円を丸ごと含む）/ 'cross'。"""
    if kind == 'disk':
        d, r = math.hypot(s['cx_nm'] - cx, s['cy_nm'] - cy), s['radius_nm']
        if d + r <= R + _EPS:
            return 'in'
        if d - r >= R - _EPS:
            return 'out'
        return 'contains' if r >= d + R - _EPS else 'cross'
    if kind == 'rect':
        x0, x1 = s['cx_nm'] - s['hw_x_nm'], s['cx_nm'] + s['hw_x_nm']
        y0, y1 = s['cy_nm'] - s['hw_y_nm'], s['cy_nm'] + s['hw_y_nm']
        far = max(math.hypot(x - cx, y - cy) for x in (x0, x1) for y in (y0, y1))
        if far <= R + _EPS:
            return 'in'
        near = math.hypot(min(max(cx, x0), x1) - cx, min(max(cy, y0), y1) - cy)
        if near >= R - _EPS:
            return 'out'
        if x0 <= cx - R + _EPS and x1 >= cx + R - _EPS and y0 <= cy - R + _EPS and y1 >= cy + R - _EPS:
            return 'contains'
        return 'cross'
    vs = s['vertices_nm']
    if max(math.hypot(x - cx, y - cy) for x, y in vs) <= R + _EPS:
        return 'in'
    dmin = min(_seg_dist(cx, cy, vs[i], vs[(i + 1) % len(vs)]) for i in range(len(vs)))
    if _point_in_polygon(cx, cy, vs):
        return 'contains' if dmin >= R - _EPS else 'cross'
    return 'out' if dmin >= R - _EPS else 'cross'


def _size(kind, s):
    """重なった形のうち「内側＝小さい方」を選ぶための大きさ（面積）。"""
    if kind == 'disk':
        return math.pi * s['radius_nm'] ** 2
    if kind == 'rect':
        return 4.0 * s['hw_x_nm'] * s['hw_y_nm']
    return abs(polygon_area(s['vertices_nm']))


# ---------------------------------------------------------------------------
# 切り抜き
# ---------------------------------------------------------------------------
def clip_layers_to_cylinder(layers, radius_nm, center=(0.0, 0.0), mode='keep', outside_material=None,
                            subst_depth_nm=0.0, tol_nm=0.05):
    """layers を円柱（中心 center、半径 radius_nm）で切り抜いた layers を返す（元の layers は変えない）。
    outside_material: 円柱の外側（mode='keep'）または穴（mode='hole'）の材質。既定は環境（'__ENV__'）。"""
    from .rcwa_s4_array2d_backend import ENV_MATERIAL
    if mode not in ('keep', 'hole'):
        raise ValueError(f"mode は 'keep' か 'hole': {mode!r}")
    R = float(radius_nm)
    if R <= 0:
        raise ValueError("切り抜く円柱の半径は 0 より大きくしてください。")
    outside = outside_material or ENV_MATERIAL
    cx, cy = float(center[0]), float(center[1])
    K = circle_polygon(cx, cy, R, tol_nm)
    src = list(layers)
    if subst_depth_nm and subst_depth_nm > 0:
        src.append({'thickness_nm': float(subst_depth_nm), 'bg_material': SUBST_MATERIAL, 'disks': []})

    out = []
    for L in src:
        shapes = ([('disk', d) for d in L.get('disks', []) if d['radius_nm'] > 0] +
                  [('rect', r) for r in L.get('rects', []) if r['hw_x_nm'] > 0 and r['hw_y_nm'] > 0] +
                  [('poly', p) for p in L.get('polygons', []) if len(p['vertices_nm']) >= 3])
        rel = [relation(k, s, cx, cy, R) for k, s in shapes]
        disks, rects, polys = [], [], []

        def keep_shape(kind, s):
            (disks if kind == 'disk' else rects if kind == 'rect' else polys).append(dict(s))

        if mode == 'keep':
            # 円の中の「下地」= 円を丸ごと含む形のうち一番内側（小さい）の材質。無ければ層の背景。
            containing = [(_size(k, s), s['material']) for (k, s), r in zip(shapes, rel) if r == 'contains']
            base = min(containing)[1] if containing else L['bg_material']
            if base != outside:
                disks.append({'material': base, 'cx_nm': cx, 'cy_nm': cy, 'radius_nm': R})
            for (kind, s), r in zip(shapes, rel):
                if r == 'in':
                    keep_shape(kind, s)
                elif r == 'cross':
                    vs = clip_to_convex(_shape_polygon(kind, s, tol_nm), K)
                    if vs:
                        polys.append({'material': s['material'], 'vertices_nm': vs})
            out.append({'thickness_nm': L['thickness_nm'], 'bg_material': outside,
                        'disks': disks, 'rects': rects, 'polygons': polys})
        else:
            for (kind, s), r in zip(shapes, rel):
                if r in ('out', 'contains'):
                    keep_shape(kind, s)
                elif r == 'cross':
                    vs = _shape_polygon(kind, s, tol_nm)
                    if not _is_convex(vs):
                        raise ValueError("凸でない多角形（レンズ状の凹凸の角の島など）が穴の縁にかかるため、"
                                         "「内側を抜く」で切り抜けません。半径か中心を少し変えてください。")
                    polys += [{'material': s['material'], 'vertices_nm': pc} for pc in subtract_convex(vs, K)]
            polys.append({'material': outside, 'vertices_nm': K})   # 穴（最後に置く = 中の形より手前）
            out.append({'thickness_nm': L['thickness_nm'], 'bg_material': L['bg_material'],
                        'disks': disks, 'rects': rects, 'polygons': polys})
    return out


def cut_structure(layers, a1, a2, spec):
    """画面の設定 spec（iridis_app/array2d_supercell_editor.py）どおりに並べて切り抜く。
    戻り値: (layers, A1, A2)。spec が None なら (layers, a1, a2) をそのまま返す。"""
    if not spec:
        return layers, np.asarray(a1, float), np.asarray(a2, float)
    nx, ny = int(spec['nx']), int(spec['ny'])
    A1, A2 = region_lattice(a1, a2, spec)
    tiled = tile_layers(layers, a1, a2, nx, ny)
    if not spec.get('cut'):
        return tiled, A1, A2
    R = float(spec['radius_nm'])
    r_in = cut_radius_limit(a1, a2, spec)
    shift = math.hypot(spec.get('cx_nm', 0.0), spec.get('cy_nm', 0.0))
    if R + shift > r_in + 1e-6:
        raise ValueError(f"切り抜く円柱（半径 {R:.1f} nm、中心のずれ {shift:.1f} nm）が上限の円"
                         f"（半径 {r_in:.1f} nm）からはみ出します（隣の周期の円柱と重なるか、並べた形の外に出る）。")
    return (clip_layers_to_cylinder(tiled, R, (spec.get('cx_nm', 0.0), spec.get('cy_nm', 0.0)),
                                    spec.get('mode', 'keep'), spec.get('outside_material'),
                                    spec.get('subst_depth_nm', 0.0)), A1, A2)

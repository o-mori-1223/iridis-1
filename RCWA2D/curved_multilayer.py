"""レンズ状の凹凸（軸対称）の上に、形に沿って積んだ多層膜を、S4 2D backend
（rcwa_s4_array2d_backend）の layers に変換する。Streamlit 非依存。

方針（2026-10-26、「階段近似＋同心円」案）:
1. 凹凸の形 h(r)（球冠・放物面・コサイン形、凸/凹）と成膜レシピから、各膜の境界面
   z_k(r) を作る（k=0 が基板表面、k=N が最上面）。膜の付き方は2通り:
   - 'normal'  : 形に垂直な方向に一定の厚さ（スパッタ・ALD 型）。下の形を半径 T_k の球で
                 なぞった上側の包絡面 z_k(x) = max_s [h(|x-s|) + sqrt(T_k^2 - s^2)]。
                 軸対称なので、点に最も近い表面の点は同じ子午面内にあり、子午面内の
                 2次元の計算で厳密に求まる。谷になる角や曲率のきついくぼみでも破綻しない。
   - 'vertical': 真上から積もる（蒸着型）。z_k = h + T_k。
   T_k は下から k 層目までの膜厚の合計。どちらの方式でも z_0 <= z_1 <= ... <= z_N。
2. 水平に輪切りにする。スライスの境目には、各境界面の平坦部の高さ・最高点・最低点を
   必ず含め、その間を厚さ dz_max 以下に等分する。
3. 各スライスの中央の高さ z_mid で、半径 r ごとの材質（z_mid より下にある境界の数で決まる）
   を求め、材質が変わる半径で「背景＋入れ子の円」にする。内容が同じ隣のスライスは
   1層にまとめる（凹凸の頂上より上の平坦な膜など）。

4. 周期構造としての扱い: 凹凸はすべて同じ形なので、各点の境界面の高さは「最も近い凹凸の
   中心からの距離 d」で決まる（凸なら厳密: 同じ形の山の包絡面の最大値は最も近い山が与える。
   凹は、隣のくぼみが最寄りの平坦部を削らない範囲で一致）。真上から見た境界は単位胞の
   Voronoi 領域（正方格子なら正方形）で区切られる。そこで、リングの半径 ρ が
   - 内接円の半径（最短格子ベクトルの半分）以下 → 円のまま
   - Voronoi 領域の外接円の半径以上 → 単位胞全体がその内側なので、背景の材質にする
   - その間 → 内側の材質を背景にし、Voronoi の頂点（単位胞の角）に残る「島」
     {d > ρ} を、周りの中心から半径 ρ の円弧で囲まれた多角形として描く
   とすれば、隣の周期像の円と重ならずに正確に表せる（膜の総厚が周期より厚くても可）。
   lattice を与えない場合は孤立した凹凸（すべて円）として扱う。

5. 多層膜円盤（disk_radius_nm を与えた場合）: 形に沿って積んだ膜を、半径 R_d の円筒で
   垂直に切り取る（成膜後にエッチング／リフトオフで円盤状に加工した形）。r > R_d では膜がなく、
   基板の表面より上は環境の材質になる。R_d は内接円の半径以下（隣の円盤と重ならない）に限る。

制限: 軸対称（真上から見て円形）のみ。
"""
import math

import numpy as np

from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL, _expand_to_bands

SHAPES = ('spherical_cap', 'paraboloid', 'cosine')
DEPOSITIONS = ('normal', 'vertical')
_DX = 0.5            # 子午面の横方向の標本間隔 [nm]
_MIN_FEATURE = 0.25  # これより細いリング・小さい円は数値誤差として捨てる [nm]
# S4 の偏光分解は、図形の法線ベクトル場を粗い格子（周期 / (resolution 8 × 次数の広がり)、
# 周期532nm・NumBasis 49 で約16nm刻み）の上で作る。それより小さい図形は法線が作れず、層の
# 固有値計算が失敗して NaN になる（実測: 半径7nmの円、大きさ1.7nmの島）。光学的には無視できる
# 大きさなので、次の大きさ未満の円・島は描かない（隣の材質に吸収させる）。
_MIN_ISLAND = 20.0   # 単位胞の角に残る島の大きさ [nm]
_MIN_RADIUS = 10.0   # 円の半径（凸の頂上・凹の底のスライスにできる）[nm]


def profile(shape, base_radius_nm, height_nm, concave=False):
    """凹凸の形 h(r) [nm]（r >= base_radius で 0）を返す。height_nm > 0。
    concave=True なら同じ形のくぼみ（-h）。球冠は高さ <= 底面の半径（半球まで）。"""
    a, H = float(base_radius_nm), float(height_nm)
    if a <= 0 or H <= 0:
        raise ValueError("底面の半径と高さは正の値にしてください")
    if shape == 'spherical_cap':
        if H > a + 1e-9:
            raise ValueError("球冠の高さは底面の半径以下にしてください（半球まで）")
        rc = (a * a + H * H) / (2 * H)               # 曲率半径
        f = lambda r: np.sqrt(np.maximum(rc * rc - r * r, 0.0)) - (rc - H)
    elif shape == 'paraboloid':
        f = lambda r: H * (1.0 - (r / a) ** 2)
    elif shape == 'cosine':
        f = lambda r: H * 0.5 * (1.0 + np.cos(np.pi * r / a))
    else:
        raise ValueError(f"unknown shape: {shape}")
    sign = -1.0 if concave else 1.0

    def h(r):
        r = np.abs(np.asarray(r, dtype=float))
        return sign * np.where(r < a, np.maximum(f(np.minimum(r, a)), 0.0), 0.0)
    return h


def film_stack(film_recipe):
    """成膜レシピ（基板段差と同じ形式、下から上）→ [(材質, 厚さ), ...]（下から上）。
    ペアは B（基板側）→ A（空気側）の順（structure_builder._expand_to_bands と同じ規約）。"""
    bands = _expand_to_bands({'subst_raise': 0.0, 'film_layers': list(film_recipe)})
    return [(mat, hi - lo) for lo, hi, mat, *_ in bands]


def influence_radius_nm(base_radius_nm, film_recipe):
    """凹凸が影響する範囲の半径（底面の半径＋膜の総厚）。これが最近接の格子点間距離の
    半分を超えると、隣の周期像の凹凸・膜と重なる。"""
    return float(base_radius_nm) + sum(t for _, t in film_stack(film_recipe))


def interfaces(h, base_radius_nm, film_recipe, deposition='normal', x_min_extent=0.0):
    """境界面の高さを子午面上の標本 x（0 <= x <= x_max）で返す。
    戻り値: (x, Z, materials)。Z は [N+1, len(x)]（Z[0] = 基板表面、Z[k] = k 層目の上面）、
    materials は下から上の膜の材質のリスト（長さ N）。"""
    if deposition not in DEPOSITIONS:
        raise ValueError(f"unknown deposition: {deposition}")
    stack = film_stack(film_recipe)
    T = np.cumsum([0.0] + [t for _, t in stack])
    x_max = max(float(base_radius_nm) + T[-1], float(x_min_extent)) + 4 * _DX
    x = np.arange(0.0, x_max + _DX / 2, _DX)
    Z = np.empty((len(T), len(x)))
    Z[0] = h(x)
    for k, Tk in enumerate(T[1:], start=1):
        if deposition == 'vertical':
            Z[k] = Z[0] + Tk
            continue
        # 半径 Tk の球（子午面では円）でなぞった包絡面。s は横方向のずれ。
        n_s = int(math.ceil(Tk / _DX))
        s = np.linspace(-Tk, Tk, 2 * n_s + 1)
        rise = np.sqrt(np.maximum(Tk * Tk - s * s, 0.0))
        zk = np.full(len(x), -np.inf)
        for chunk in np.array_split(np.arange(len(s)), max(1, len(s) // 256)):
            cand = h(x[None, :] - s[chunk, None]) + rise[chunk, None]
            np.maximum(zk, cand.max(axis=0), out=zk)
        Z[k] = zk
    return x, Z, [m for m, _ in stack]


def _slice_heights(Z, dz_max, x=None, disk_radius=None):
    crit = set()
    for zk in Z:
        crit.update((round(float(zk[-1]), 9), round(float(zk.max()), 9), round(float(zk.min()), 9)))
        if disk_radius is not None:   # 円盤の側面での各境界の高さ（側面の縞を正しく切る）
            crit.add(round(float(np.interp(disk_radius, x, zk)), 9))
    crit = sorted(crit)
    edges = [crit[0]]
    for lo, hi in zip(crit[:-1], crit[1:]):
        n = max(1, int(math.ceil((hi - lo) / dz_max - 1e-9)))
        edges += list(np.linspace(lo, hi, n + 1)[1:])
    return np.array(edges)


def _rings_at(x, Z, z_mid, names, disk_radius=None):
    """高さ z_mid での半径方向の材質分布 → (外側の材質, [(半径, 内側の材質), ...] 外→内)。
    disk_radius を与えると、その半径より外の膜を取り除く（基板より上は環境）。"""
    idx = (Z <= z_mid).sum(axis=0)               # 0=基板, 1..N=膜, N+1=環境
    if disk_radius is not None:
        outside = x > disk_radius
        idx = np.where(outside, np.where(Z[0] <= z_mid, len(Z), 0), idx)
    bg = names[idx[-1]]
    change = np.nonzero(np.diff(idx))[0]          # idx[i] != idx[i+1]
    rings = []
    for i in change[::-1]:                        # 外側から
        k_in, k_out = idx[i], idx[i + 1]
        if disk_radius is not None and x[i] <= disk_radius < x[i + 1] and k_in != 0 and k_in <= len(Z) - 1:
            rings.append((float(disk_radius), names[k_in]))   # 円盤の側面（垂直）で膜が切れる
            continue
        # 境界の半径を、z_mid を横切る境界面の線形補間で求める
        k = min(k_in, k_out)                      # 横切っている境界面の番号
        z0, z1 = Z[k, i] - z_mid, Z[k, i + 1] - z_mid
        t = z0 / (z0 - z1) if z0 != z1 else 0.5
        r = x[i] + float(np.clip(t, 0.0, 1.0)) * (x[i + 1] - x[i])
        rings.append((r, names[k_in]))
    # 細すぎるリング（数値誤差）を除いて、同じ材質が続くものをまとめる
    cleaned = []
    for r, mat in rings:
        if r < _MIN_RADIUS:
            continue
        if cleaned and cleaned[-1][0] - r < _MIN_FEATURE:
            cleaned[-1] = (r, mat)
            continue
        cleaned.append((r, mat))
    out, outer = [], bg
    for r, mat in cleaned:
        if mat != outer:
            out.append((r, mat))
            outer = mat
    return bg, out


class _LatticeGeometry:
    """格子（a1, a2 [nm]）の Voronoi 領域: 内接円・外接円の半径と、単位胞あたりの頂点。"""
    _ISLAND_SAMPLES = 72

    def __init__(self, a1, a2):
        from scipy.spatial import Voronoi
        a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
        rng = range(-3, 4)
        self.points = np.array([i * a1 + j * a2 for i in rng for j in rng])
        origin = int(np.argmin(np.linalg.norm(self.points, axis=1)))
        vor = Voronoi(self.points)
        verts = vor.vertices[[v for v in vor.regions[vor.point_region[origin]] if v >= 0]]
        self.r_in = 0.5 * min(float(np.linalg.norm(p)) for p in self.points if np.linalg.norm(p) > 1e-9)
        self.r_cov = float(np.linalg.norm(verts, axis=1).max())
        # 単位胞あたりの頂点（格子の並進で重なるものは1つにまとめる）
        inv = np.linalg.inv(np.column_stack([a1, a2]))
        uniq, self.vertices = set(), []
        for v in verts:
            frac = np.mod(inv @ v, 1.0)
            key = tuple(np.round(frac, 6) % 1.0)
            if key not in uniq:
                uniq.add(key)
                self.vertices.append(v)

    def island_size(self, rho):
        """単位胞の角に残る島 {d > rho} の大きさ（縦横の広がりの最大、全頂点のうち最大）。"""
        if rho <= self.r_in:
            return np.inf
        return max(float(np.ptp(np.array(self.island(v, rho)), axis=0).max()) for v in self.vertices)

    def island(self, v, rho):
        """頂点 v のまわりの {d > rho}（d = 最も近い格子点までの距離）を多角形で返す（反時計回り）。"""
        w = v[None, :] - self.points                      # 各格子点から見た v
        ww = (w * w).sum(axis=1)
        pts = []
        for th in np.linspace(0, 2 * np.pi, self._ISLAND_SAMPLES, endpoint=False):
            u = np.array([np.cos(th), np.sin(th)])
            b = w @ u
            disc = b * b - (ww - rho * rho)
            t = np.where(disc >= 0, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
            t = np.where(t > 0, t, np.inf)
            pts.append(v + t.min() * u)
        return pts


def _slice_shapes(bg, rings, geom, cx, cy):
    """1枚のスライスのリング（外→内）→（背景, 円, 多角形）。geom=None なら全部円。"""
    if geom is None:
        return bg, [{'material': m, 'cx_nm': cx, 'cy_nm': cy, 'radius_nm': r} for r, m in rings], []
    rings = list(rings)
    # 外接円以上のリング: 単位胞全体がその内側 → 背景を内側の材質にする。外接円にごく近く、
    # 角に残る島が _MIN_ISLAND 未満になるリングも同様に扱う（実測: 大きさ1.7nmの島を含む層で
    # S4 の層の固有値計算が失敗した。スライス厚より桁違いに小さく、物理的には無視できる）。
    while rings and (rings[0][0] >= geom.r_cov - _MIN_FEATURE or geom.island_size(rings[0][0]) < _MIN_ISLAND):
        bg = rings.pop(0)[1]
    big = [(r, m) for r, m in rings if r > geom.r_in]          # 外→内
    small = [(r, m) for r, m in rings if r <= geom.r_in]
    polygons = []
    if big:
        outer = [bg] + [m for _, m in big[:-1]]                # 各リングの外側の材質
        bg = big[-1][1]                                        # 最も内側の大リングの内側
        for (rho, _m_in), m_out in reversed(list(zip(big, outer))):   # 内→外の順に重ねる
            for v in geom.vertices:
                polygons.append({'material': m_out,
                                 'vertices_nm': [(cx + p[0], cy + p[1]) for p in geom.island(v, rho)]})
    disks = [{'material': m, 'cx_nm': cx, 'cy_nm': cy, 'radius_nm': r} for r, m in small]
    return bg, disks, polygons


def build_layers(shape, base_radius_nm, height_nm, film_recipe, concave=False,
                 deposition='normal', dz_max_nm=10.0, cx_nm=0.0, cy_nm=0.0, lattice=None,
                 disk_radius_nm=None):
    """凹凸＋多層膜 → rcwa_s4_array2d_backend 用の layers（top-first）。
    lattice=(a1, a2)（nm）を与えると周期構造として扱う（4. 参照）。
    disk_radius_nm を与えると、膜を半径 R_d の円盤に切り取る（多層膜円盤、5. 参照）。
    戻り値: (layers, info)。info はプレビュー用（x, Z, スライス境界, 各スライスのリング）。"""
    geom = _LatticeGeometry(*lattice) if lattice is not None else None
    rd = None if disk_radius_nm is None else float(disk_radius_nm)
    if rd is not None:
        if rd < _MIN_RADIUS:
            raise ValueError(f"円盤の半径は {_MIN_RADIUS:.0f} nm 以上にしてください")
        if geom is not None and rd > geom.r_in + 1e-6:
            raise ValueError(f"円盤の半径（{rd:.0f} nm）が最近接の格子点間距離の半分"
                             f"（{geom.r_in:.0f} nm）を超え、隣の円盤と重なります")
    h = profile(shape, base_radius_nm, height_nm, concave)
    x, Z, mats = interfaces(h, base_radius_nm, film_recipe, deposition, x_min_extent=rd or 0.0)
    names = [SUBST_MATERIAL] + mats + [ENV_MATERIAL]
    edges = _slice_heights(Z, float(dz_max_nm), x, rd)

    slices = []   # 下から: (z_lo, z_hi, bg, rings)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 1e-9:
            continue
        bg, rings = _rings_at(x, Z, 0.5 * (lo + hi), names, rd)
        if slices and slices[-1][2] == bg and len(slices[-1][3]) == len(rings) and all(
                m1 == m2 and abs(r1 - r2) < _MIN_FEATURE for (r1, m1), (r2, m2) in zip(slices[-1][3], rings)):
            slices[-1] = (slices[-1][0], hi, bg, slices[-1][3])   # 同じ内容のスライスはまとめる
        else:
            slices.append((lo, hi, bg, rings))

    layers = []
    for lo, hi, bg, rings in reversed(slices):
        lbg, disks, polygons = _slice_shapes(bg, rings, geom, float(cx_nm), float(cy_nm))
        layers.append({'thickness_nm': hi - lo, 'bg_material': lbg, 'disks': disks, 'polygons': polygons})
    info = {'x': x, 'Z': Z, 'materials': mats, 'slices': slices, 'disk_radius_nm': rd,
            'influence_radius_nm': influence_radius_nm(base_radius_nm, film_recipe),
            'r_in': geom.r_in if geom else None, 'r_cov': geom.r_cov if geom else None}
    return layers, info

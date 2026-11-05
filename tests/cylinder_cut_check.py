"""周期領域（スーパーセル）と円柱の切り抜き（RCWA2D/cylinder_cut.py）の検証。

形の計算:
 1) 内接円の半径（正方 3×3 = 1.5a、六方 3×3 = 3a√3/4、余白つき）
 2) 円と円・矩形と円の交わりの面積が理論値（数値積分）と合う
 3) 穴あけ: 形から円を引いたピースは重ならず、面積の合計 = 形 − 交わり
 4) 形と円の関係の判定（中・外・境界・円を丸ごと含む）
S4 での計算:
 5) 1×1 の平坦膜を「内側を残す」で切り抜く = 同じ半径の円盤層（全桁一致）
 6) 1×1 の平坦膜を「内側を抜く」で穴あけ = 背景が膜で円が環境の円盤層（穴は多角形なので差は小さいが 0 ではない）
 7) 基板にも食い込ませる（深さ d）= 基板材質の円柱を d の厚さで足した構造（全桁一致）
 8) 並べただけ（切り抜きなし）は元の周期構造と同じ（2×2、NumBasis を 4 倍）
 9) 3×3 の円盤を、境界にかかる円盤ができる半径で切り抜く: R+T≈1、多角形の細かさ（tol）を変えても結果がほぼ同じ
10) 「内側を抜く」で境界にかかる円盤: R+T≈1

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/cylinder_cut_check.py
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

import RCWA2D.rcwa_s4_array2d_backend as b2
from RCWA2D import cylinder_cut as cc
from RCWA2D.material_spec import resolve_nk_fn

ENV, SUB = b2.ENV_MATERIAL, b2.SUBST_MATERIAL
WL = np.array([450.0, 550.0, 650.0])
FMAP = {'TiO2': resolve_nk_fn('TiO2'), 'SiO2': resolve_nk_fn('SiO2')}
A = 400.0
a1, a2 = np.array([A, 0.0]), np.array([0.0, A])


def solve(layers, A1, A2, norder=49, theta_deg=0.0):
    px, py, gamma = cc._unit_pitch(A1, A2)
    return b2.calc_array_rcwa_s4(WL, np.radians(theta_deg), 0.0, px / 1000.0, py / 1000.0, norder, layers, FMAP,
                                 resolve_nk_fn(1.0), resolve_nk_fn(1.45), gamma_deg=gamma)


def grid_area(inside_fn, box, n=1500):
    x = np.linspace(box[0], box[1], n); y = np.linspace(box[2], box[3], n)
    X, Y = np.meshgrid(x, y)
    return inside_fn(X, Y).mean() * (box[1] - box[0]) * (box[3] - box[2])


def main():
    # 1) 内接円
    A1, A2 = cc.supercell_vectors(a1, a2, 3, 3)
    assert abs(cc.inscribed_radius(A1, A2) - 1.5 * A) < 1e-9
    h1, h2 = np.array([A, 0.0]), np.array([A / 2, A * math.sqrt(3) / 2])
    H1, H2 = cc.supercell_vectors(h1, h2, 3, 3)
    assert abs(cc.inscribed_radius(H1, H2) - 3 * A * math.sqrt(3) / 4) < 1e-9
    G1, G2 = cc.supercell_vectors(a1, a2, 3, 3, gap_nm=100.0)
    assert abs(cc.inscribed_radius(G1, G2) - (1.5 * A + 50.0)) < 1e-9
    print(f"[1] 内接円: 正方3×3 {cc.inscribed_radius(A1, A2):.1f} nm、六方3×3 {cc.inscribed_radius(H1, H2):.1f} nm、"
          f"余白100nm {cc.inscribed_radius(G1, G2):.1f} nm")

    # 2) 交わりの面積
    R, d, r = 400.0, 300.0, 200.0
    K = cc.circle_polygon(0, 0, R)
    lens = cc.clip_to_convex(cc.circle_polygon(d, 0, r), K)
    exact = (r * r * math.acos((d * d + r * r - R * R) / (2 * d * r)) + R * R * math.acos((d * d + R * R - r * r) / (2 * d * R))
             - 0.5 * math.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)))
    # 円は内接正多角形（頂点が円周上、弦と円弧のずれ ≤ 0.05 nm）なので面積は少し小さい
    # （半径が約 0.03 nm 小さいのに相当。頂点を円周上に置くのは、切り取ったピースが円からはみ出さないため）
    assert abs(cc.polygon_area(lens) - exact) / exact < 1e-3, (cc.polygon_area(lens), exact)
    rect = [(250.0, -150.0), (550.0, -150.0), (550.0, 150.0), (250.0, 150.0)]
    rk = cc.clip_to_convex(rect, K)
    ref = grid_area(lambda X, Y: X ** 2 + Y ** 2 <= R * R, (250, 550, -150, 150))
    assert abs(cc.polygon_area(rk) - ref) / ref < 2e-3, (cc.polygon_area(rk), ref)
    print(f"[2] 円∩円 {cc.polygon_area(lens):.1f} nm²（理論 {exact:.1f}）、矩形∩円 {cc.polygon_area(rk):.0f}（数値積分 {ref:.0f}）")

    # 3) 穴あけのピース
    for S in (cc.circle_polygon(d, 0, r), rect, [(-500.0, -60.0), (500.0, -60.0), (500.0, 60.0), (-500.0, 60.0)]):
        pieces = cc.subtract_convex(S, K)
        tot = sum(cc.polygon_area(p) for p in pieces)
        want = cc.polygon_area(S) - cc.polygon_area(cc.clip_to_convex(S, K))
        assert abs(tot - want) / want < 1e-9, (tot, want)
        rng = np.random.default_rng(0)
        pts = rng.uniform(-600, 600, size=(20000, 2))
        cover = sum(np.array([cc._point_in_polygon(x, y, p) for x, y in pts]) for p in pieces)
        assert cover.max() <= 1, "ピースが重なっている"
    print(f"[3] 穴あけ: ピースの面積の合計 = 形 − 交わり、重なりなし（横長の矩形は {len(cc.subtract_convex(S, K))} ピース・左右 2 つに分かれる）")

    # 4) 関係の判定
    rel = cc.relation
    assert rel('disk', {'cx_nm': 0, 'cy_nm': 0, 'radius_nm': 100}, 0, 0, 150) == 'in'
    assert rel('disk', {'cx_nm': 400, 'cy_nm': 0, 'radius_nm': 100}, 0, 0, 150) == 'out'
    assert rel('disk', {'cx_nm': 0, 'cy_nm': 0, 'radius_nm': 300}, 0, 0, 150) == 'contains'
    assert rel('disk', {'cx_nm': 150, 'cy_nm': 0, 'radius_nm': 100}, 0, 0, 150) == 'cross'
    assert rel('rect', {'cx_nm': 0, 'cy_nm': 0, 'hw_x_nm': 500, 'hw_y_nm': 500}, 0, 0, 150) == 'contains'
    assert rel('rect', {'cx_nm': 0, 'cy_nm': 0, 'hw_x_nm': 500, 'hw_y_nm': 60}, 0, 0, 150) == 'cross'
    print("[4] 形と円の関係（中・外・円を含む・境界）")

    # 5) 1×1 の平坦膜を切り抜く = 円盤層
    film = [{'thickness_nm': 100.0, 'bg_material': 'TiO2', 'disks': []}]
    cut = cc.clip_layers_to_cylinder(film, 150.0)
    disk = [{'thickness_nm': 100.0, 'bg_material': ENV,
             'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}]
    d5 = np.max(np.abs(solve(cut, a1, a2)['R0'] - solve(disk, a1, a2)['R0']))
    assert d5 < 1e-12, d5
    print(f"[5] 1×1 平坦膜を R=150 で切り抜く = 円盤層 r=150: max|ΔR0| = {d5:.1e}")

    # 6) 穴あけ = 背景 TiO2 に環境の円
    hole = cc.clip_layers_to_cylinder(film, 150.0, mode='hole')
    ring = [{'thickness_nm': 100.0, 'bg_material': 'TiO2',
             'disks': [{'material': ENV, 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}]
    d6 = np.max(np.abs(solve(hole, a1, a2)['R0'] - solve(ring, a1, a2)['R0']))
    assert d6 < 1e-3, d6
    print(f"[6] 1×1 平坦膜に R=150 の穴（{len(hole[0]['polygons'][0]['vertices_nm'])} 角形）= 円の穴: max|ΔR0| = {d6:.1e}")

    # 7) 基板にも食い込ませる
    deep = cc.clip_layers_to_cylinder(film, 150.0, subst_depth_nm=50.0)
    ref7 = disk + [{'thickness_nm': 50.0, 'bg_material': ENV,
                    'disks': [{'material': SUB, 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}]
    d7 = np.max(np.abs(solve(deep, a1, a2)['R0'] - solve(ref7, a1, a2)['R0']))
    assert d7 < 1e-12, d7
    print(f"[7] 基板に 50 nm 食い込ませる = 基板材質の円柱を足した構造: max|ΔR0| = {d7:.1e}")

    # 8) 並べただけ = 元の周期構造
    disk120 = [{'thickness_nm': 100.0, 'bg_material': ENV,
                'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 120.0}]}]
    t2 = cc.tile_layers(disk120, a1, a2, 2, 2)
    B1, B2 = cc.supercell_vectors(a1, a2, 2, 2)
    one = solve(disk120, a1, a2, norder=49, theta_deg=20.0)
    two = solve(t2, B1, B2, norder=4 * 49, theta_deg=20.0)
    d8 = np.max(np.abs(one['R0'] - two['R0']))
    assert len(t2[0]['disks']) == 4 and d8 < 2e-3, d8
    print(f"[8] 2×2 に並べただけ（NumBasis 196）vs 1×1（49）、θ=20°: max|ΔR0| = {d8:.1e}  R_total {np.round(two['R_total'], 4)}")

    # 9) 3×3 を境界にかかる半径で切り抜く
    t3 = cc.tile_layers(disk120, a1, a2, 3, 3)
    C1, C2 = cc.supercell_vectors(a1, a2, 3, 3)
    # 中央は中、辺の中央の 4 個（中心から 400 nm）は境界にかかり、角の 4 個（566 nm、内側の端 446 nm）は外
    R9 = 420.0
    rels = [cc.relation('disk', dd, 0, 0, R9) for dd in t3[0]['disks']]
    assert rels.count('in') == 1 and rels.count('cross') == 4 and rels.count('out') == 4, rels
    c9 = cc.clip_layers_to_cylinder(t3, R9)
    c9f = cc.clip_layers_to_cylinder(t3, R9, tol_nm=0.01)
    r9, r9f = solve(c9, C1, C2, norder=121), solve(c9f, C1, C2, norder=121)
    rt = r9['R_total'] + r9['T_total']
    assert np.all(np.abs(rt - 1) < 0.02), rt
    d9 = np.max(np.abs(r9['R0'] - r9f['R0']))
    assert d9 < 1e-4, d9
    print(f"[9] 3×3・R=420（中 1・境界 4・外 4）: R+T = {rt.min():.4f}〜{rt.max():.4f}、"
          f"多角形の細かさ 0.05→0.01 nm で max|ΔR0| = {d9:.1e}")

    # 10) 穴あけで境界にかかる円盤
    h10 = cc.clip_layers_to_cylinder(t3, R9, mode='hole')
    r10 = solve(h10, C1, C2, norder=121)
    rt10 = r10['R_total'] + r10['T_total']
    assert np.all(np.abs(rt10 - 1) < 0.02), rt10
    print(f"[10] 3×3・R=420 の穴（境界にかかる円盤は {sum(1 for p in h10[0]['polygons'] if p['material'] == 'TiO2')} ピース）: "
          f"R+T = {rt10.min():.4f}〜{rt10.max():.4f}")
    print("ALL OK")


if __name__ == "__main__":
    main()

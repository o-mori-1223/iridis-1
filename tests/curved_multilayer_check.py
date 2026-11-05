"""レンズ状の凹凸＋形に沿った多層膜（RCWA2D/curved_multilayer.py）の検証。

1) 平坦の極限: 凹凸の高さ→0 で、平面多層膜と一致する
2) 膜なしの極限: 膜厚0・高さ=半径の球冠は、既存の半球層（同じスライス数）と一致する
3) なだらかな形では、2つの膜の付き方（形に垂直／真上から）が一致する
4) 損失のない材質でエネルギー保存 R+T≈1（凸・凹とも）
5) スライス厚さ Δz の収束（10nm → 5nm の変化）
6) 画面操作（凹＋多層膜円盤）〜計算
7) 多層膜円盤: 平坦の極限で、既存の多層膜円盤（各層が円盤）と一致する
8) 多層膜円盤のエネルギー保存・隣と重なる半径はエラー

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/curved_multilayer_check.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

import RCWA2D.rcwa_s4_array2d_backend as b2
from RCWA2D import s4_parallel
from RCWA2D.curved_multilayer import build_layers, film_stack
from RCWA2D.material_spec import resolve_nk_fn

RECIPE = [{'type': 'simple', 'material': 'TiO2', 'thickness': 51.0},
          {'type': 'pair', 'mat_a': 'TiO2', 'th_a': 51.0, 'mat_b': 'SiO2', 'th_b': 81.0, 'n_pairs': 3}]
FMAP = {'TiO2': resolve_nk_fn('TiO2'), 'SiO2': resolve_nk_fn('SiO2')}
VAC, SUB = resolve_nk_fn(1.0), resolve_nk_fn('SiO2')
WL = np.array([450.0, 550.0, 650.0])
PITCH = 0.532
SQUARE = (np.array([532.0, 0.0]), np.array([0.0, 532.0]))
HEX = (np.array([532.0, 0.0]), np.array([266.0, 532.0 * np.sqrt(3) / 2]))


def build(*args, lattice=SQUARE, **kw):
    return build_layers(*args, lattice=lattice, **kw)


def solve(layers, wl=WL, theta_deg=0.0, norder=49):
    return b2.calc_array_rcwa_s4(wl, np.radians(theta_deg), 0.0, PITCH, PITCH, norder, layers, FMAP, VAC, SUB)


def main():
    # 1) 平坦の極限
    layers, _ = build('spherical_cap', 150.0, 1e-6, RECIPE)
    flat =[{'thickness_nm': t, 'bg_material': m, 'disks': []} for m, t in reversed(film_stack(RECIPE))]
    a, b = solve(layers, theta_deg=20.0), solve(flat, theta_deg=20.0)
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[1] 平坦の極限: 平面多層膜との差 max|ΔR0| = {d:.1e}")
    assert d < 1e-4

    # 2) 膜なしの極限（半球、Δz=7.5nm → 20スライス）
    layers, _ = build('spherical_cap', 150.0, 150.0, [], dz_max_nm=7.5)
    hemi = b2.hemisphere_disk_layers(150.0, b2.SUBST_MATERIAL, 20)
    a, b = solve(layers), solve(hemi)
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[2] 膜なしの極限: 半球層（20スライス）との差 max|ΔR0| = {d:.1e}（スライス数 {len(layers)}）")
    assert len(layers) == 20 and d < 1e-3

    # 3) なだらかな形（高さ5nm）では膜の付き方によらない
    ln, _ = build('paraboloid', 150.0, 5.0, RECIPE, deposition='normal', dz_max_nm=5.0)
    lv, _ = build('paraboloid', 150.0, 5.0, RECIPE, deposition='vertical', dz_max_nm=5.0)
    a, b = solve(ln), solve(lv)
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[3] なだらかな形: 形に垂直 vs 真上から 差 max|ΔR0| = {d:.1e}")
    assert d < 5e-3

    # 4) エネルギー保存（凸・凹、2方式、正方・六方格子、斜め入射も）。膜の総厚(441nm)は
    #    周期の半分より厚く、単位胞の角の「島」（多角形）を使う構造になっている。
    th = np.radians([0.0, 30.0])
    for shape, concave, dep, lat, gamma in (
            ('spherical_cap', False, 'normal', SQUARE, 90.0), ('spherical_cap', True, 'normal', SQUARE, 90.0),
            ('cosine', True, 'vertical', SQUARE, 90.0), ('paraboloid', False, 'vertical', SQUARE, 90.0),
            ('spherical_cap', False, 'normal', HEX, 60.0)):
        layers, _ = build(shape, 150.0, 100.0, RECIPE, concave=concave, deposition=dep, lattice=lat)
        n_poly = sum(len(L['polygons']) for L in layers)
        r = s4_parallel.solve_2d(WL, th, 0.0, PITCH, PITCH, 49, layers, FMAP, VAC, SUB, gamma_deg=gamma)
        rt = r['R_total'] + r['T_total']
        print(f"[4] {shape:13s} {'凹' if concave else '凸'} {dep:8s} {'六方' if gamma == 60 else '正方'}: "
              f"スライス {len(layers):3d}層 島 {n_poly:2d}個  R+T = {rt.min():.4f}〜{rt.max():.4f}  "
              f"NaN {int(np.isnan(rt).sum())}")
        assert not np.isnan(rt).any() and np.all(np.abs(rt - 1) < 0.01)

    # 5) Δz の収束
    l10, _ = build('spherical_cap', 150.0, 100.0, RECIPE, dz_max_nm=10.0)
    l5, _ = build('spherical_cap', 150.0, 100.0, RECIPE, dz_max_nm=5.0)
    a = s4_parallel.solve_2d(WL, np.array([0.0]), 0.0, PITCH, PITCH, 49, l10, FMAP, VAC, SUB)
    b = s4_parallel.solve_2d(WL, np.array([0.0]), 0.0, PITCH, PITCH, 49, l5, FMAP, VAC, SUB)
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[5] Δz 10nm({len(l10)}層) → 5nm({len(l5)}層): max|ΔR0| = {d:.1e}")
    assert d < 0.02

    # 7) 多層膜円盤: 平坦の極限で、既存の「多層膜円盤」（各層が円盤、周りは環境）と一致する
    layers, _ = build('spherical_cap', 150.0, 1e-6, RECIPE, disk_radius_nm=200.0)
    flat_disk = [{'thickness_nm': t, 'bg_material': b2.ENV_MATERIAL,
                  'disks': [{'material': m, 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 200.0}]}
                 for m, t in reversed(film_stack(RECIPE))]
    a, b = solve(layers, theta_deg=20.0), solve(flat_disk, theta_deg=20.0)
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[7] 多層膜円盤・平坦の極限: 既存の多層膜円盤との差 max|ΔR0| = {d:.1e}")
    assert d < 1e-3

    # 8) 多層膜円盤のエネルギー保存（凸・凹、円盤の半径 = 底面の半径 / 内接円いっぱい）
    for shape, concave, dep, lat, gamma, rd in (
            ('spherical_cap', False, 'normal', SQUARE, 90.0, 150.0),
            ('spherical_cap', True, 'normal', SQUARE, 90.0, 250.0),
            ('cosine', False, 'vertical', HEX, 60.0, 266.0)):
        layers, _ = build(shape, 150.0, 100.0, RECIPE, concave=concave, deposition=dep, lattice=lat,
                          disk_radius_nm=rd)
        r = s4_parallel.solve_2d(WL, th, 0.0, PITCH, PITCH, 49, layers, FMAP, VAC, SUB, gamma_deg=gamma)
        rt = r['R_total'] + r['T_total']
        print(f"[8] 多層膜円盤 R={rd:.0f} {shape:13s} {'凹' if concave else '凸'} {dep:8s} "
              f"{'六方' if gamma == 60 else '正方'}: スライス {len(layers):3d}層  R+T = {rt.min():.4f}〜{rt.max():.4f}")
        assert not np.isnan(rt).any() and np.all(np.abs(rt - 1) < 0.01)
    try:
        build('spherical_cap', 150.0, 100.0, RECIPE, disk_radius_nm=300.0)
        raise AssertionError("隣と重なる円盤の半径でエラーにならない")
    except ValueError:
        print("[8] 隣と重なる円盤の半径（300nm > 266nm）はエラーになる")

    # 6) 画面: レンズ状の凹凸を有効化 → 断面図・3Dプレビュー → 凹に切替 → 計算
    import time
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=180)
    at.run()
    at.radio(key='app_mode').set_value('2次元周期アレイ (2D)').run()
    next(b for b in at.button if b.key and b.key.startswith('al_rm_')).click().run()   # 既定の円盤層を削除
    {n.label: n for n in at.number_input}["ピッチ x [nm]"].set_value(532.0).run()
    at.checkbox(key='lens_enabled').check().run()
    assert not at.exception, at.exception
    assert any('S4 に渡す凹凸部分' in c.value for c in at.caption)
    at.session_state["a2d_tab"] = "🔍 形状プレビュー"; at.run()   # 形状プレビューのタブを開く
    assert not at.exception, at.exception
    at.radio(key='lens_concave').set_value("凹（くぼみ）").run()
    assert not at.exception, at.exception
    at.radio(key='lens_extent').set_value('disk').run()        # 多層膜円盤（R=150nm）
    assert not at.exception, at.exception
    assert "円盤の半径 R [nm]" in [n.label for n in at.number_input]
    {n.label: n for n in at.number_input}["NumBasis（次数）"].set_value(25).run()
    {n.label: n for n in at.number_input}["波長点数"].set_value(5).run()
    {b.label: b for b in at.button}["▶ 計算を実行"].click().run()
    for _ in range(600):
        th = at.session_state["calc_state"].get("thread")
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
        at.run()
    at.run()
    cs = at.session_state["calc_state"]
    assert not at.exception and cs["error"] is None and cs["result"] is not None, (at.exception, cs["error"])
    assert not np.isnan(cs["result"]["R0"]).any()
    rt = cs["result"]["R_total"] + cs["result"]["T_total"]
    print(f"[6] 画面: 凹のレンズ状構造＋多層膜円盤の計算 R0={np.round(cs['result']['R0'], 3)}  R+T {rt.min():.4f}〜{rt.max():.4f}")
    assert np.all(np.abs(rt - 1) < 0.02)
    print("ALL OK")


if __name__ == "__main__":
    main()

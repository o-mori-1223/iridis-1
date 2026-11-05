"""横倒しシリンダー（2Dモードの層 'hcylinder'、rcwa_s4_array2d_backend.hcylinder_rect_layers）の検証。

1) 階段近似: 総高さ = 直径、体積 → πr²L（n_slices を増やすと収束）
2) 回転対称: 正方格子で「x 軸のシリンダー・方位角 0°」と「y 軸のシリンダー・方位角 90°」は
   構造と入射を同時に 90° 回したものなので、R0 が一致する
3) 無限に長い棒（周期いっぱい）: x 方向に一様なので、垂直入射では x 方向の周期を変えても
   結果が変わらない（端が隣の周期像とぴったりつながっていることの確認）
4) エネルギー保存 R+T≈1（有限長・無限長、垂直・斜め入射、損失のない材質）
5) n_slices の収束（20 → 40）
6) 外形が隣の周期像と重なる判定、断面図（y 方向に切ると丸い断面が出る）
7) 画面: 横倒しシリンダーを追加 → プレビュー（y 方向の断面）→ 計算

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/hcylinder_check.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

import RCWA2D.rcwa_s4_array2d_backend as b2
from RCWA2D import s4_parallel
from RCWA2D.material_spec import resolve_nk_fn
from iridis_app.array2d_cross_section import _materials_at
from iridis_app.array2d_model import build_layers, hcylinder_overlaps

FMAP = {'TiO2': resolve_nk_fn('TiO2')}
VAC, SUB = resolve_nk_fn(1.0), resolve_nk_fn('SiO2')
WL = np.array([450.0, 550.0, 650.0])
SQ = (np.array([532.0, 0.0]), np.array([0.0, 532.0]))


def cyl(axis='x', infinite=False, r=100.0, length=400.0, n=20, **kw):
    return dict({'type': 'hcylinder', 'material': 'TiO2', 'bg_material': b2.ENV_MATERIAL, 'radius_nm': r,
                 'length_nm': length, 'axis': axis, 'infinite': infinite, 'n_slices': n,
                 'cx_nm': 0.0, 'cy_nm': 0.0}, **kw)


def layers_of(spec, lat=SQ):
    return build_layers([spec], lattice=lat)[0]


def solve(layers, px=0.532, py=0.532, theta_deg=0.0, phi_deg=0.0, norder=49):
    return b2.calc_array_rcwa_s4(WL, np.radians(theta_deg), np.radians(phi_deg), px, py, norder, layers,
                                 FMAP, VAC, SUB)


def main():
    # 1) 階段近似
    for n in (20, 200):
        L = b2.hcylinder_rect_layers(100.0, 400.0, 'x', 'TiO2', n)
        h = sum(l['thickness_nm'] for l in L)
        vol = sum(l['thickness_nm'] * 4 * l['rects'][0]['hw_x_nm'] * l['rects'][0]['hw_y_nm'] for l in L)
        err = vol / (np.pi * 100.0 ** 2 * 400.0) - 1
        print(f"[1] n_slices={n:3d}: 総高さ {h:.1f} nm、体積の誤差 {err:+.2e}")
        assert abs(h - 200.0) < 1e-9 and abs(err) < (0.01 if n == 20 else 2e-4)

    # 2) 回転対称（斜め入射でも）
    for th in (0.0, 25.0):
        a = solve(layers_of(cyl('x')), theta_deg=th, phi_deg=0.0)
        b = solve(layers_of(cyl('y')), theta_deg=th, phi_deg=90.0)
        d = np.max(np.abs(a['R0'] - b['R0']))
        print(f"[2] 回転対称 θ={th:.0f}°: x軸・φ=0° vs y軸・φ=90° max|ΔR0| = {d:.1e}  (R0={np.round(a['R0'], 4)})")
        assert d < 1e-6

    # 3) 無限に長い棒: x 方向の周期によらない。E が棒に垂直（φ=0°）は次数の収束が遅いが、
    #    reduce_uniform_period で全次数を棒に垂直な方向に回すので、NumBasis 49 でも収束値に近い
    lat300 = (np.array([300.0, 0.0]), np.array([0.0, 532.0]))
    a = solve(layers_of(cyl('x', infinite=True)), px=0.532)
    b = solve(layers_of(cyl('x', infinite=True), lat300), px=0.300)
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[3] 無限に長い棒: x方向の周期 532 vs 300 nm  max|ΔR0| = {d:.1e}  (R0={np.round(a['R0'], 4)})")
    assert d < 1e-9
    ref = solve(layers_of(cyl('x', infinite=True)), norder=321)
    d = np.max(np.abs(a['R0'] - ref['R0']))
    print(f"[3] 無限に長い棒（E⊥棒）: NumBasis 49 と 321 の差 max|ΔR0| = {d:.1e}")
    assert d < 5e-3
    for phi in (0.0, 90.0):   # 縮める前（周期532nmのまま、1次元の次数が少ない）との比較のため、斜め入射でも
        a = solve(layers_of(cyl('x', infinite=True)), theta_deg=30.0, phi_deg=phi)
        b = solve(layers_of(cyl('x', infinite=True), lat300), px=0.300, theta_deg=30.0, phi_deg=phi)
        assert np.max(np.abs(a['R0'] - b['R0'])) < 1e-9
    print("[3] 斜め入射（θ=30°、φ=0°/90°）でも x 方向の周期によらない")

    # 4) エネルギー保存
    th = np.radians([0.0, 30.0])
    for name, spec in (('有限長 x軸 L=400', cyl('x')), ('有限長 y軸 L=300 r=150', cyl('y', r=150.0, length=300.0)),
                       ('無限長 x軸', cyl('x', infinite=True)), ('無限長 y軸', cyl('y', infinite=True))):
        r = s4_parallel.solve_2d(WL, th, 0.0, 0.532, 0.532, 49, layers_of(spec), FMAP, VAC, SUB)
        rt = r['R_total'] + r['T_total']
        print(f"[4] {name:22s}: R+T = {rt.min():.4f}〜{rt.max():.4f}  NaN {int(np.isnan(rt).sum())}")
        assert not np.isnan(rt).any() and np.all(np.abs(rt - 1) < 0.01)

    # 5) n_slices の収束
    a, b = solve(layers_of(cyl('x', n=20))), solve(layers_of(cyl('x', n=40)))
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[5] n_slices 20 → 40: max|ΔR0| = {d:.1e}")
    assert d < 0.01

    # 6) 重なり判定・断面
    assert not hcylinder_overlaps(cyl('x', length=532.0), *SQ) and hcylinder_overlaps(cyl('x', length=540.0), *SQ)
    assert not hcylinder_overlaps(cyl('x', infinite=True, r=266.0), *SQ)
    assert hcylinder_overlaps(cyl('x', r=270.0), *SQ)
    HEX = (np.array([532.0, 0.0]), np.array([266.0, 532.0 * np.sqrt(3) / 2]))
    assert hcylinder_overlaps(cyl('x', length=500.0, r=240.0), *HEX)        # 斜めの隣と重なる
    try:
        layers_of(cyl('y', infinite=True), HEX)
        raise AssertionError("六方格子で y 方向の無限長がエラーにならない")
    except ValueError:
        pass
    L_all = layers_of(cyl('x'))
    ys = np.linspace(-266.0, 266.0, 533)
    shifts = [tuple(i * SQ[0] + j * SQ[1]) for i in range(-3, 4) for j in range(-3, 4)]
    widths = [float(np.sum(_materials_at(L, np.zeros_like(ys), ys, shifts) == 'TiO2')) for L in L_all]
    print(f"[6] 重なり判定 OK。y 方向の断面での幅（上→下、中央付近が最大 ≈ 直径）: "
          f"{widths[0]:.0f} … {max(widths):.0f} … {widths[-1]:.0f} nm")
    assert max(widths) >= 195 and widths[0] < 90 and widths[-1] < 90

    # 7) 画面
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=180)
    at.run()
    at.radio(key='app_mode').set_value('2次元周期アレイ (2D)').run()
    next(b for b in at.button if b.key and b.key.startswith('al_rm_')).click().run()
    {n.label: n for n in at.number_input}["ピッチ x [nm]"].set_value(532.0).run()
    {b.label: b for b in at.button}["＋ 横倒しシリンダー"].click().run()
    assert not at.exception, at.exception
    at.session_state["a2d_tab"] = "🔍 形状プレビュー"; at.run()   # 形状プレビューのタブを開く
    at.radio(key='xsec_dir').set_value("y 方向に切る").run()
    assert not at.exception, at.exception
    next(c for c in at.checkbox if c.key and c.key.startswith('al_inf_')).check().run()
    assert not at.exception, at.exception
    {n.label: n for n in at.number_input}["NumBasis（次数）"].set_value(25).run()
    {n.label: n for n in at.number_input}["波長点数"].set_value(5).run()
    {b.label: b for b in at.button}["▶ 計算を実行"].click().run()
    for _ in range(600):
        t = at.session_state["calc_state"].get("thread")
        if t is None or not t.is_alive():
            break
        time.sleep(0.5)
        at.run()
    at.run()
    cs = at.session_state["calc_state"]
    assert not at.exception and cs["error"] is None and cs["result"] is not None, (at.exception, cs["error"])
    rt = cs["result"]["R_total"] + cs["result"]["T_total"]
    print(f"[7] 画面: 無限に長い横倒しシリンダーの計算 R0={np.round(cs['result']['R0'], 3)}  "
          f"R+T {rt.min():.4f}〜{rt.max():.4f}")
    assert not np.isnan(rt).any() and np.all(np.abs(rt - 1) < 0.02)
    print("ALL OK")


if __name__ == "__main__":
    main()

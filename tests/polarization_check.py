"""2Dモードの偏光角 ψ（0 = s 偏光、90 = p 偏光、その間は直線偏光）の検証。

1) 平坦な単層膜で、ψ=0°/90° が解析解（薄膜の Fresnel 公式）の s/p 反射率と一致する
2) 垂直入射では「ψ=90°・φ=0°」=「ψ=0°・φ=90°」（電場の向きが同じ）
3) 直線偏光の重ね合わせ: (R(ψ)+R(−ψ))/2 = cos²ψ·R(s) + sin²ψ·R(p)（どんな構造でも厳密）
4) エネルギー保存 R+T≈1（p 偏光・斜め入射、損失のない材質）
5) 画面: サイドバーで ψ=90° にして計算 → 結果が s 偏光と違い、タイトルに p偏光と出る

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/polarization_check.py
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

N_FILM, N_SUB, D_FILM = 2.3, 1.45, 120.0
WL = np.array([450.0, 550.0, 650.0])


def thin_film_R(wl, theta_deg, pol):
    """空気 / 膜（n=2.3, 120nm）/ 基板（n=1.45）の反射率（解析解）。"""
    n0, n1, n2 = 1.0, N_FILM, N_SUB
    s0 = np.sin(np.radians(theta_deg))
    c = [np.sqrt(1 - (n0 * s0 / n) ** 2 + 0j) for n in (n0, n1, n2)]
    def r(na, ca, nb, cb):
        return (na * ca - nb * cb) / (na * ca + nb * cb) if pol == 's' else (nb * ca - na * cb) / (nb * ca + na * cb)
    r01, r12 = r(n0, c[0], n1, c[1]), r(n1, c[1], n2, c[2])
    beta = 2 * np.pi / wl * n1 * D_FILM * c[1]
    rr = (r01 + r12 * np.exp(2j * beta)) / (1 + r01 * r12 * np.exp(2j * beta))
    return np.abs(rr) ** 2


def solve(layers, fmap, theta_deg=0.0, phi_deg=0.0, pol_deg=0.0, norder=49):
    return b2.calc_array_rcwa_s4(WL, np.radians(theta_deg), np.radians(phi_deg), 0.532, 0.532, norder, layers,
                                 fmap, resolve_nk_fn(1.0), resolve_nk_fn(N_SUB), pol_deg=pol_deg)


def main():
    # 1) 解析解
    film = [{'thickness_nm': D_FILM, 'bg_material': 'F', 'disks': []}]
    fmap = {'F': resolve_nk_fn(N_FILM)}
    for th in (0.0, 40.0, 70.0):
        for pol, psi in (('s', 0.0), ('p', 90.0)):
            d = np.max(np.abs(solve(film, fmap, th, 30.0, psi)['R0'] - thin_film_R(WL, th, pol)))
            print(f"[1] 平坦な単層膜 θ={th:.0f}° {pol}偏光: 解析解との差 {d:.1e}")
            assert d < 1e-9
    rs, rp = thin_film_R(WL, 70.0, 's'), thin_film_R(WL, 70.0, 'p')
    print(f"    θ=70°: 解析解 Rs={np.round(rs, 3)} Rp={np.round(rp, 3)}（p の方が小さい＝ブリュースター角に近い）")

    # 2), 3) 横倒しシリンダー（x 軸）
    cyl = b2.hcylinder_rect_layers(100.0, 400.0, 'x', 'TiO2', 20)
    fmap = {'TiO2': resolve_nk_fn('TiO2')}
    a, b = solve(cyl, fmap, 0.0, 0.0, 90.0), solve(cyl, fmap, 0.0, 90.0, 0.0)
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[2] 垂直入射: ψ=90°・φ=0° vs ψ=0°・φ=90° max|ΔR0| = {d:.1e}")
    assert d < 1e-9
    s, p = solve(cyl, fmap, 30.0, 0.0, 0.0), solve(cyl, fmap, 30.0, 0.0, 90.0)
    print(f"[3] θ=30°: R0(s)={np.round(s['R0'], 4)}  R0(p)={np.round(p['R0'], 4)}")
    assert np.max(np.abs(s['R0'] - p['R0'])) > 1e-3
    # 直線偏光の重ね合わせ: R(ψ) = cos²ψ·Rs + sin²ψ·Rp + 2cosψ·sinψ·(交差項)。交差項は ψ の符号で
    # 反転するので、(R(ψ)+R(-ψ))/2 = cos²ψ·Rs + sin²ψ·Rp はどんな構造でも厳密に成り立つ。
    # 鏡映対称な構造では交差項は 0 のはずだが、S4 の偏光分解（法線ベクトル場を格子上で作る）が
    # 対称性をわずかに崩し、NumBasis 49 で ~1e-3 残る（偏光分解なしなら 1e-15、NumBasis を増やすと減る。
    # 2026-10-29 実測）。ここでは厳密な恒等式を確かめ、交差項の大きさは参考として表示する。
    for psi in (30.0, 45.0, 60.0):
        mp, mm = solve(cyl, fmap, 30.0, 0.0, psi), solve(cyl, fmap, 30.0, 0.0, -psi)
        c2 = np.cos(np.radians(psi)) ** 2
        for key in ('R0', 'R_total', 'T_total'):
            d = np.max(np.abs(0.5 * (mp[key] + mm[key]) - (c2 * s[key] + (1 - c2) * p[key])))
            assert d < 1e-9, (psi, key, d)
        if psi == 45.0:
            print(f"[3] 交差項（R0(+45°)−R0(−45°)、鏡映対称なので本来 0）= {np.max(np.abs(mp['R0'] - mm['R0'])):.1e}"
                  "（S4 の偏光分解の離散化による。NumBasis を増やすと減る）")
    print("[3] (R(ψ)+R(−ψ))/2 = cos²ψ·R(s) + sin²ψ·R(p)（ψ=30/45/60°、R0・R_total・T_total）")

    # 4) エネルギー保存
    for name, layers in (('横倒しシリンダー', cyl),
                         ('円盤', [{'thickness_nm': 100.0, 'bg_material': b2.ENV_MATERIAL,
                                   'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}])):
        r = s4_parallel.solve_2d(WL, np.radians([0.0, 30.0, 60.0]), np.radians(20.0), 0.532, 0.532, 49, layers,
                                 fmap, resolve_nk_fn(1.0), resolve_nk_fn(N_SUB), pol_deg=90.0)
        rt = r['R_total'] + r['T_total']
        print(f"[4] {name} p偏光・φ=20°・θ=0/30/60°: R+T = {rt.min():.4f}〜{rt.max():.4f}")
        assert np.all(np.abs(rt - 1) < 0.015)

    # 5) 画面
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=180)
    at.run()
    at.radio(key='app_mode').set_value('2次元周期アレイ (2D)').run()
    {n.label: n for n in at.number_input}["NumBasis（次数）"].set_value(25).run()
    {n.label: n for n in at.number_input}["波長点数"].set_value(5).run()
    {n.label: n for n in at.number_input}["入射角 θ [deg]"].set_value(40.0).run()

    def run_calc():
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
        return np.array(cs["result"]["R0"]), cs

    r_s, _ = run_calc()
    at.slider(key='pol_deg').set_value(90.0).run()
    r_p, cs = run_calc()
    import json
    title = json.loads(at.get('plotly_chart')[-1].proto.spec)['layout']['title']['text']
    print(f"[5] 画面: θ=40° R0(s)={np.round(r_s, 3)}  R0(p)={np.round(r_p, 3)}  pol_deg={cs['pol_deg']}")
    assert cs['pol_deg'] == 90.0 and np.max(np.abs(r_s - r_p)) > 1e-3 and 'p偏光' in title
    print("ALL OK")


if __name__ == "__main__":
    main()

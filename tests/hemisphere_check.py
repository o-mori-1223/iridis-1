"""2Dモードの半球（平らな面を下にしたドーム形）層の確認。

1) 形状: 半球(n枚)の staircase は、球(2n枚)の上半分と厳密に同じ（同じ中点則のため）
2) 物理: TiO2 半球アレイで R+T≈1（無損失）、n_slices を増やすと R0 が収束する
3) 画面: 「＋ 半球層を追加」→ 3Dプレビュー → 計算 が例外なく通る

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/hemisphere_check.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from streamlit.testing.v1 import AppTest

import RCWA2D.rcwa_s4_array2d_backend as b2
from RCWA2D.material_spec import resolve_nk_fn


def main():
    # 1) 形状
    hemi = b2.hemisphere_disk_layers(150.0, 'TiO2', 10)
    sph = b2.sphere_disk_layers(150.0, 'TiO2', 20)
    assert len(hemi) == 10
    for h, s in zip(hemi, sph[:10]):
        assert abs(h['thickness_nm'] - s['thickness_nm']) < 1e-12
        assert abs(h['disks'][0]['radius_nm'] - s['disks'][0]['radius_nm']) < 1e-9
    assert abs(sum(L['thickness_nm'] for L in hemi) - 150.0) < 1e-9
    print("[OK] 半球(10枚) = 球(20枚)の上半分、総高さ = 半径")

    # 2) 物理
    wl = np.array([450.0, 550.0, 650.0])
    fmap = {'TiO2': resolve_nk_fn('TiO2')}
    prev = None
    for n in (4, 10, 30, 80):
        r = b2.calc_array_rcwa_s4(wl, 0.0, 0.0, 0.532, 0.532, 49, b2.hemisphere_disk_layers(150.0, 'TiO2', n),
                                  fmap, resolve_nk_fn(1.0), resolve_nk_fn('SiO2'))
        rt = r['R_total'] + r['T_total']
        print(f"     n_slices={n:>2}: R0={np.round(r['R0'], 4)}  R+T={np.round(rt, 4)}")
        assert np.all(np.abs(rt - 1) < 0.01), "エネルギー保存則が崩れている"
        if n == 80:
            assert np.max(np.abs(r['R0'] - prev)) < 5e-3, "n_slices 30→80 で R0 が収束していない"
        prev = r['R0']
    print("[OK] エネルギー保存（|R+T-1|<1%）・n_slices 収束")

    # 3) 画面
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=120)
    at.run()
    at.radio(key='app_mode').set_value('2次元周期アレイ (2D)').run()
    next(b for b in at.button if b.key and b.key.startswith('al_rm_')).click().run()   # 既定の円盤層を削除
    {b.label: b for b in at.button}["＋ 半球層を追加"].click().run()
    assert not at.exception, at.exception
    assert any('半球の高さ' in c.value for c in at.caption)
    at.session_state["a2d_tab"] = "🔍 形状プレビュー"; at.run()   # 形状プレビューのタブを開く
    assert not at.exception, at.exception
    nis = {n.label: n for n in at.number_input}
    nis["NumBasis（次数）"].set_value(25).run()
    {n.label: n for n in at.number_input}["波長点数"].set_value(5).run()
    {b.label: b for b in at.button}["▶ 計算を実行"].click().run()
    for _ in range(240):
        th = at.session_state["calc_state"].get("thread")
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
        at.run()
    at.run()
    cs = at.session_state["calc_state"]
    assert not at.exception and cs["error"] is None and cs["result"] is not None
    assert not np.isnan(cs["result"]["R0"]).any()
    print(f"[OK] 画面: 半球層の追加・プレビュー・計算（R0={np.round(cs['result']['R0'], 4)}）")
    print("ALL OK")


if __name__ == "__main__":
    main()

"""iridis.py 2Dモード（2次元周期アレイ）のスモーク＋回帰テスト。
計算はバックグラウンドスレッドで走るので、終わるまで再実行してポーリングする。

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/app_smoke_2d.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from streamlit.testing.v1 import AppTest

# 2026-10-26 の分割前（iridis.py 単一ファイル版）で得た値（既定の円盤層＋追加の円盤層、
# NumBasis=25、5波長点、θ=0°）。
REF_R0_FIRST = 0.09176071107693994


def check(at, label):
    assert not at.exception, f"[{label}] 例外: {[str(e) for e in at.exception]}"
    print(f"[OK] {label}")


def run_calc(at, label):
    {b.label: b for b in at.button}["▶ 計算を実行"].click().run(timeout=120)
    check(at, label + "（実行ボタン）")
    for _ in range(240):
        th = at.session_state["calc_state"].get("thread")
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
        at.run(timeout=120)
    at.run(timeout=120)
    check(at, label + "（結果表示）")
    cs = at.session_state["calc_state"]
    assert cs["error"] is None, f"計算スレッド内でエラー: {cs['error']}"
    assert cs["result"] is not None, "計算結果が None（スレッドの書き込みが見えていない?）"
    return cs


def main():
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=120)
    at.run()
    at.radio(key='app_mode').set_value('2次元周期アレイ (2D)').run()
    check(at, "2Dモードへ切替")
    {b.label: b for b in at.button}["＋ 円盤層を追加"].click().run()
    check(at, "円盤層を追加")
    at.session_state["a2d_tab"] = "🔍 形状プレビュー"; at.run()   # 形状プレビューのタブを開く
    check(at, "3D形状プレビュー")
    nis = {n.label: n for n in at.number_input}
    nis["NumBasis（次数）"].set_value(25).run()
    nis = {n.label: n for n in at.number_input}
    nis["波長点数"].set_value(5).run()

    cs = run_calc(at, "単一角度の計算")
    r0 = cs["result"]["R0"]
    assert cs["result"]["n_computed"] == 5 and r0.shape == (5,) and not np.isnan(r0).any()
    assert abs(r0[0] - REF_R0_FIRST) < 1e-10, f"R0[0] が基準値と違う: {r0[0]}"
    print(f"     R0[0] = {r0[0]}（基準値と一致）")

    {r.label: r for r in at.radio}["入射角モード"].set_value("角度掃引").run()
    {s.label: s for s in at.slider}["入射角 θ の範囲 [deg]"].set_value((0.0, 40.0)).run()
    {n.label: n for n in at.number_input}["角度ステップ [deg]"].set_value(20.0).run()
    cs = run_calc(at, "角度掃引（0/20/40°）＋スウォッチ")
    wl = cs["wl_ar"]
    for target in (400.0, 470.0, 540.0, 600.0, 700.0):
        assert np.any(np.isclose(wl, target)), f"波長グリッドに{target}nmが無い"
    assert list(cs["theta_deg_ar"]) == [0.0, 20.0, 40.0]
    assert cs["result"]["R0"].shape == (3, len(wl)) and cs["result"]["n_computed"] == 3 * len(wl)
    assert cs["swatch_result"]["R0"].shape == (3, len(wl))
    assert abs(cs["result"]["R0"][0, 0] - REF_R0_FIRST) < 1e-10
    print(f"     R0.shape = {cs['result']['R0'].shape}, swatch {cs['swatch_result']['R0'].shape}")
    print("ALL OK")


if __name__ == "__main__":
    main()

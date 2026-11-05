"""array2d_app.py（legacy、iridis.pyの2次元周期アレイモードに統合済み）の
スモークテスト（streamlit.testing.v1.AppTestで、実際にブラウザを開かずに
ボタン操作を再現し、例外なく動くか確認する）。

回帰テスト用に恒久的に残す。NumBasisを小さく(9)・波長点数も少なく
しているため、S4があれば数秒〜十数秒で終わる。

Run: source ~/s4env/bin/activate && cd <repo> && python3 legacy/diag_array2d_app_smoke.py

【設計メモ】array2d_app.pyは計算をバックグラウンドスレッドで実行し、
`threading.Event`で途中停止できるようにしている。スレッドは
`st.session_state`に直接書き込まず、普通のPython dict
（session_stateに一度だけ格納した参照）を経由する設計にしている——
バックグラウンドスレッドから`st.session_state.x = y`のように直接書くと、
（このテストで最初に発見した実際の不具合）ScriptRunContextが無いスレッドからの
書き込みが呼び出し側から見えなくなることがあったため。このテストは
その回帰も検出できるよう、実行後に`calc_state["result"]`が実際に取得できる
ことを確認する。

2026-09-25追加: 球形状(shape_mode)・角度掃引(angle_mode)のシナリオも
カバーする（新規追加された機能の回帰チェック）。
"""
import time

import numpy as np
from streamlit.testing.v1 import AppTest

APP_PATH = __file__.replace("diag_array2d_app_smoke.py", "array2d_app.py")


def sget(at, key, default=None):
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


def _run_and_wait(at, label):
    """「計算を実行」ボタンを押し、スレッドが終わるまでポーリングする。"""
    btns = {b.label: b for b in at.button}
    btns["▶ 計算を実行"].click().run(timeout=60)
    assert not at.exception, f"[{label}] 計算実行ボタンで例外: {at.exception}"

    for _ in range(40):
        cs = sget(at, "calc_state")
        th = cs["thread"] if cs else None
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
        at.run(timeout=60)
        assert not at.exception, f"[{label}] ポーリング中に例外: {at.exception}"

    cs = sget(at, "calc_state")
    assert cs is not None, f"[{label}] calc_stateが存在しない"
    assert cs["error"] is None, f"[{label}] 計算スレッド内でエラー: {cs['error']}"
    assert cs["result"] is not None, (
        f"[{label}] 計算結果がNone -- バックグラウンドスレッドの書き込みが"
        "st.session_state経由で見えていない可能性(regression!)"
    )
    return cs


def main():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception, f"初期表示で例外: {at.exception}"
    print("[OK] 初期表示")

    btns = {b.label: b for b in at.button}
    btns["🔍 形状を表示（計算前プレビュー）"].click().run(timeout=60)
    assert not at.exception, f"形状表示ボタンで例外: {at.exception}"
    print("[OK] 形状表示ボタン")

    nis = {ni.label: ni for ni in at.number_input}
    nis["NumBasis（次数）"].set_value(9).run(timeout=60)
    assert not at.exception

    btns = {b.label: b for b in at.button}
    btns["⏱ 所要時間を見積もる（1点だけ試し計算）"].click().run(timeout=120)
    assert not at.exception, f"時間見積もりボタンで例外: {at.exception}"
    print("[OK] 所要時間見積もりボタン")

    nis = {ni.label: ni for ni in at.number_input}
    nis["波長点数"].set_value(5).run(timeout=60)
    assert not at.exception

    print("=== シナリオ1: 円盤・単一角度（既存機能） ===")
    cs = _run_and_wait(at, "disk/single")
    assert cs["angle_mode"] == "単一角度"
    assert cs["result"]["n_computed"] == 5
    print(f"[OK] n_computed={cs['result']['n_computed']}, R0[0]={cs['result']['R0'][0]:.4f}")

    print("=== シナリオ2: 球・単一角度（新規: 形状=球） ===")
    radios = {r.label: r for r in at.radio}
    radios["形状"].set_value("球 (Sphere)").run(timeout=60)
    assert not at.exception, f"形状=球への切替で例外: {at.exception}"
    nis = {ni.label: ni for ni in at.number_input}
    assert "staircase分割数 n_slices" in nis, "球モードでn_slices入力が出ていない"
    nis["staircase分割数 n_slices"].set_value(4).run(timeout=60)  # 高速化のため粗くする
    assert not at.exception
    btns = {b.label: b for b in at.button}
    btns["🔍 形状を表示（計算前プレビュー）"].click().run(timeout=60)
    assert not at.exception, f"球の形状表示で例外: {at.exception}"
    cs = _run_and_wait(at, "sphere/single")
    assert cs["result"]["n_computed"] == 5
    print(f"[OK] n_computed={cs['result']['n_computed']}, R0[0]={cs['result']['R0'][0]:.4f}")

    print("=== シナリオ3: 円盤・角度掃引（新規: 入射角モード=角度掃引） ===")
    radios = {r.label: r for r in at.radio}
    radios["形状"].set_value("円盤 (Disk)").run(timeout=60)
    assert not at.exception
    radios = {r.label: r for r in at.radio}
    radios["入射角モード"].set_value("角度掃引").run(timeout=60)
    assert not at.exception, f"角度掃引への切替で例外: {at.exception}"
    sliders = {s.label: s for s in at.slider}
    assert "入射角 θ の範囲 [deg]" in sliders, "角度掃引モードでθ範囲スライダーが出ていない"
    sliders["入射角 θ の範囲 [deg]"].set_value((0.0, 40.0)).run(timeout=60)
    assert not at.exception
    nis = {ni.label: ni for ni in at.number_input}
    nis["角度ステップ [deg]"].set_value(20.0).run(timeout=60)  # 0,20,40度の3点だけに絞って高速化
    assert not at.exception
    cs = _run_and_wait(at, "disk/sweep")
    assert cs["angle_mode"] == "角度掃引"
    n_theta_expected = 3  # 0,20,40度

    # 波長グリッド(380-780nmを5点)には元々400/470/540/600/700が乗らないため、
    # Gsolver'S COLOR.html互換の角度チャート用に自動で合流されているはず
    # (2026-09-25の修正——合流前は5点、合流後は5+5=10点になる)。
    wl_ar = cs["wl_ar"]
    for target in (400.0, 470.0, 540.0, 600.0, 700.0):
        assert np.any(np.isclose(wl_ar, target)), (
            f"波長グリッドに{target}nmが厳密に含まれていない: {wl_ar}"
        )
    n_wl_expected = len(wl_ar)
    assert n_wl_expected == 10, f"合流後の波長点数が想定と違う: {n_wl_expected} (grid={wl_ar})"
    print(f"[OK] 波長グリッドに400/470/540/600/700nmが厳密に合流済み (計{n_wl_expected}点)")

    assert cs["result"]["n_computed"] == n_theta_expected * n_wl_expected, (
        f"n_computed={cs['result']['n_computed']}, 期待={n_theta_expected * n_wl_expected}"
    )
    assert cs["result"]["R0"].shape == (n_theta_expected, n_wl_expected)
    selects = {sb.label: sb for sb in at.selectbox}
    assert "表示する量" in selects, "角度掃引の結果表示にヒートマップの量選択が出ていない"
    print(f"[OK] n_computed={cs['result']['n_computed']}, "
          f"R0.shape={cs['result']['R0'].shape}")

    # 度→ラジアン→度の往復で29.999999999999996のような表示汚れが出ないか確認
    # (2026-09-25の修正——「計算を実行」時に保存した元の度配列をそのまま使う)。
    theta_deg_ar = cs["theta_deg_ar"]
    assert list(theta_deg_ar) == [0.0, 20.0, 40.0], (
        f"theta_deg_arが厳密な値になっていない(浮動小数点誤差の疑い): {list(theta_deg_ar)!r}"
    )
    print(f"[OK] theta_deg_arが厳密な値: {list(theta_deg_ar)}")

    # Gsolver'S COLOR.html互換表示(render_gsolver_style_display)用の
    # スウォッチ(0/30/60度)追加計算がワーカースレッド内で完走しているか確認。
    swatch = cs["swatch_result"]
    assert swatch is not None, "swatch_resultが計算されていない(角度掃引でのスウォッチ計算漏れ)"
    assert swatch["R0"].shape == (3, n_wl_expected), f"swatch shape異常: {swatch['R0'].shape}"
    assert swatch["n_computed"] == 3 * n_wl_expected, (
        f"swatch n_computed={swatch['n_computed']}, 期待={3 * n_wl_expected}"
    )
    assert not at.exception, f"Gsolver互換表示の描画で例外: {at.exception}"
    print(f"[OK] swatch_result計算完了、R0.shape={swatch['R0'].shape} "
          f"(render_gsolver_style_display呼び出しも例外なし)")

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()

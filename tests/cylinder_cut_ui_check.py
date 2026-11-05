"""2Dモードの「周期領域と円柱の切り抜き」の画面の検証（AppTest）。

 1) オンにして 3×3 にすると、計算する周期が 1200×1200 nm と表示される
 2) 「NumBasis を … にする」で NumBasis が 49×9=441 になる
 3) 形状プレビュー（断面図・3D）がエラーなく出る。断面図は切り抜いた形、3D は周期領域と円柱の壁
 4) 計算すると R+T≈1。計算に使った周期が 1200 nm、解析記録用の条件に supercell が入る
 5) 半径を手で内接円より大きくするとエラーが出て、計算に進まない
 6) 解析記録の条件から restore_2d で入力欄を戻せる（3×3・穴あけ・手入力の半径）

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/cylinder_cut_ui_check.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from streamlit.testing.v1 import AppTest

from iridis_app.records_restore import restore_2d


def check(at, label):
    assert not at.exception, f"[{label}] 例外: {[str(e) for e in at.exception]}"
    print(f"[OK] {label}")


def main():
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=600)
    at.run()
    at.radio(key="app_mode").set_value("2次元周期アレイ (2D)").run()
    {n.label: n for n in at.number_input}["波長点数"].set_value(3).run()

    # 1) オン・3×3
    at.checkbox(key="sc_enabled").check().run()
    at.radio(key="sc_preset").set_value("3×3").run()
    assert any("計算する周期: 1200 nm × 1200 nm（3×3 周期）" in c.value for c in at.caption), [c.value for c in at.caption]
    assert any("S4 に渡す層" in c.value and "多角形" in c.value for c in at.caption)
    check(at, "[1] 3×3 にすると計算する周期が 1200×1200 nm（角の円盤が円柱の境界にかかり多角形になる）")

    # 2) NumBasis の目安
    at.button(key="sc_set_nb").click().run()
    assert at.session_state["norder"] == 441, at.session_state["norder"]
    check(at, "[2] 「NumBasis を 441 にする」（49 × 9）")
    {n.label: n for n in at.number_input}["NumBasis（次数）"].set_value(81).run()   # テストは軽くする

    # 3) プレビュー
    at.session_state["a2d_tab"] = "🔍 形状プレビュー"; at.run()   # 形状プレビューのタブを開く
    import json
    def titles():
        return [json.loads(c.proto.spec)['layout']['title']['text'] for c in at.get('plotly_chart')]
    assert len(titles()) == 2, titles()   # 左に断面、右に 3D
    assert any("切り抜いた形の 3D モデル" in t for t in titles()), titles()
    assert any("3D モデルをダウンロード" in b.label for b in at.get("download_button")), "OBJ のボタンがない"
    # AppTest はタブの選択を送らないので、実行のたびに session_state でタブを指定する
    at.radio(key="preview_3d_kind").set_value("切り抜く前＋円柱の壁")
    at.session_state["a2d_tab"] = "🔍 形状プレビュー"; at.run()
    assert any("3×3 周期を表示" in t and "円柱" in t for t in titles()), titles()
    at.session_state["a2d_tab"] = "🧱 構造"; at.run()
    assert not at.get('plotly_chart'), "開いていないタブの図を描いている"
    check(at, "[3] 形状プレビュー（断面図・切り抜いた形の 3D モデル・OBJ、切り替えで周期領域と円柱の壁。閉じると描かない）")

    # 4) 計算
    {b.label: b for b in at.button}["▶ 計算を実行"].click().run()
    for _ in range(600):
        th = at.session_state["calc_state"].get("thread")
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
        at.run()
    at.run()
    cs = at.session_state["calc_state"]
    assert cs["error"] is None and cs["result"] is not None, cs["error"]
    rt = np.array(cs["result"]["R_total"]) + np.array(cs["result"]["T_total"])
    assert np.all(np.abs(rt - 1) < 0.03), rt
    sc = cs["record"]["params"]["supercell"]
    assert sc["nx"] == 3 and sc["cut"] and sc["mode"] == "keep" and abs(sc["radius_nm"] - 599.0) < 1e-9
    assert "3×3 周期・円柱 R=599 nm で切り抜き" in cs["record"]["text"]
    assert any("3×3 周期" in c.value and "R_total" in c.value for c in at.caption)
    check(at, f"[4] 計算: R+T = {rt.min():.4f}〜{rt.max():.4f}、記録の条件に supercell（R=599 nm）")

    # 5) 大きすぎる半径
    at.checkbox(key="sc_r_auto").uncheck().run()
    at.number_input(key="sc_r").set_value(700.0).run()
    assert any("はみ出し" in e.value for e in at.error), [e.value for e in at.error]
    check(at, "[5] 半径 700 nm（内接円 600 nm より大きい）でエラー")

    # 6) 記録からの読み込み
    ss = {}
    params = {"sidebar": {"lattice_shape": "正方格子", "pitch_x_nm": 400.0, "angle_mode": "単一角度"},
              "array_layers": [], "step_regions": None, "lens": None,
              "supercell": {"nx": 3, "ny": 3, "cut": True, "mode": "hole", "outside_material": "__ENV__",
                            "gap_nm": 0.0, "radius_nm": 450.0, "r_auto": False, "cx_nm": 10.0, "cy_nm": 0.0,
                            "subst_depth_nm": 20.0}}
    restore_2d(ss, params, ["TiO2", "SiO2"])
    assert ss["sc_enabled"] and ss["sc_preset"] == "3×3" and ss["sc_mode"] == "hole" and ss["sc_out"] == "環境と同じ"
    assert not ss["sc_r_auto"] and ss["sc_r"] == 450.0 and ss["sc_cx"] == 10.0 and ss["sc_depth"] == 20.0
    ss2 = {}
    restore_2d(ss2, dict(params, supercell=None), ["TiO2"])
    assert ss2["sc_enabled"] is False
    print("[OK] [6] restore_2d で周期領域の設定が戻る（無い記録ではオフ）")

    # 7) 切り抜いた形の 3D モデル（画面を使わない部分）
    from RCWA2D import cylinder_cut as cc
    from iridis_app.array2d_cut_preview import build_cut_model, cut_model_obj

    def volume(v, t):   # 閉じたメッシュの体積（発散定理）
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        return abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()) / 6.0

    a1, a2 = np.array([400.0, 0.0]), np.array([0.0, 400.0])
    film = [{'thickness_nm': 100.0, 'bg_material': 'TiO2', 'disks': []}]
    g, _, top, _ = build_cut_model(cc.clip_layers_to_cylinder(film, 150.0), a1, a2, draw_substrate=False)
    (mat, op, v, t), = list(g.items())
    vol = volume(v, t)
    # 描画用の円は正 n 角形（頂点が円周上）なので、体積は n 角柱の値と比べる（円柱より 0.3% ほど小さい）
    n = len(cc.circle_polygon(0, 0, 150.0, 0.5, min_n=48))
    want = 0.5 * n * 150.0 ** 2 * np.sin(2 * np.pi / n) * 100.0
    assert mat == 'TiO2' and op == 1.0 and abs(vol / want - 1) < 1e-9 and top == 100.0, (vol, want)
    disk = [{'thickness_nm': 100.0, 'bg_material': '__ENV__',
             'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 120.0}]}]
    t3 = cc.tile_layers(disk + film, a1, a2, 3, 3)
    cut3 = cc.clip_layers_to_cylinder(t3, 420.0)
    g3, _, _, _ = build_cut_model(cut3, *cc.supercell_vectors(a1, a2, 3, 3), draw_substrate=False)
    rmax = max(np.hypot(vv[:, 0], vv[:, 1]).max() for _, _, vv, _ in g3.items())
    assert rmax <= 420.0 + 0.6, rmax   # 描画用の円は弦と円弧のずれ ≤ 0.5 nm
    obj = cut_model_obj(cut3, *cc.supercell_vectors(a1, a2, 3, 3), "SiO2")
    rows = obj.splitlines()
    assert "g TiO2" in rows and sum(r.startswith("v ") for r in rows) > 0 and sum(r.startswith("f ") for r in rows) > 0
    nv = sum(r.startswith("v ") for r in rows)
    assert max(max(int(x) for x in r.split()[1:]) for r in rows if r.startswith("f ")) == nv   # 面の番号が頂点数の範囲内
    print(f"[OK] [7] 3D モデル: 1×1 平坦膜を R=150 で切り抜くと {n} 角柱の体積と一致（円柱の {vol / (np.pi * 150 ** 2 * 100):.4f} 倍）、"
          f"3×3・R=420 の切り抜きで形が円柱からはみ出さない（最大半径 {rmax:.2f} nm）、OBJ {obj.count(chr(10))} 行")
    print("ALL OK")


if __name__ == "__main__":
    main()

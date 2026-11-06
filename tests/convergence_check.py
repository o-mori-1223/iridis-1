"""2Dモードの「🧪 収束確認」（iridis_app/array2d_convergence.py）の検証。

1) 補助関数: NumBasis の既定の列、カンマ区切りの読み取り、階段近似を2倍細かくする
   （球・半球・横倒しシリンダー・ペア内・レンズの Δz）
2) 画面: 横倒しシリンダーで収束確認を実行 → NumBasis の各設定＋階段2倍の行が出て、判定が表示される。
   「階段2倍」の行の R0 は、同じ NumBasis で n_slices を2倍にして直接解いた値と一致する

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/convergence_check.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

from iridis_app.array2d_convergence import _MAX_NUMBASIS, default_numbasis_list, parse_numbers
from iridis_app.array2d_model import build_layers, refined_inputs
from RCWA2D.structure_builder import ENV_MATERIAL


def main():
    assert default_numbasis_list(49) == [49, 98, 196]
    assert default_numbasis_list(301) == [301, 602, min(1204, _MAX_NUMBASIS)]
    assert parse_numbers("450, 550、650 x") == [450.0, 550.0, 650.0]
    assert parse_numbers("25 49", int) == [25, 49]

    cyl = {'type': 'hcylinder', 'material': 'TiO2', 'bg_material': ENV_MATERIAL, 'radius_nm': 100.0,
           'length_nm': 400.0, 'axis': 'x', 'infinite': False, 'n_slices': 20, 'cx_nm': 0.0, 'cy_nm': 0.0}
    film = {'type': 'film', 'material': 'SiO2', 'thickness_nm': 50.0}
    pair = {'type': 'pair', 'n_pairs': 2, 'a': dict(cyl, type='sphere'), 'b': film}
    out, lens = refined_inputs([cyl, film, pair])
    assert out[0]['n_slices'] == 40 and out[2]['a']['n_slices'] == 40 and out[1] == film and lens is None
    assert cyl['n_slices'] == 20                                   # 元は書き換えない
    assert refined_inputs([film]) is None
    _, lens = refined_inputs([film], ({'dz_max_nm': 10.0}, []))
    assert lens[0]['dz_max_nm'] == 5.0
    print("[1] 補助関数 OK")

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=180)
    at.run()
    at.radio(key='app_mode').set_value('2次元周期アレイ (2D)').run()
    next(b for b in at.button if b.key and b.key.startswith('al_rm_')).click().run()
    {n.label: n for n in at.number_input}["ピッチ x [nm]"].set_value(532.0).run()
    {b.label: b for b in at.button}["＋ 横倒しシリンダー"].click().run()
    {n.label: n for n in at.number_input}["NumBasis（次数）"].set_value(25).run()
    at.text_input(key='conv_wls').set_value("450, 650").run()
    at.text_input(key='conv_nbs').set_value("25, 49").run()
    assert at.checkbox(key='conv_slices').value                     # 階段近似の要素があるので既定でオン
    {b.label: b for b in at.button}["🧪 収束を確認"].click().run()
    for _ in range(600):
        th = at.session_state["conv_state"].get("thread")
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
        at.run()
    at.run()
    cs = at.session_state["conv_state"]
    assert not at.exception and cs['error'] is None, (at.exception, cs['error'])
    labels = [r['label'] for r in cs['rows']]
    print(f"[2] 画面: 収束確認の行 {labels}")
    assert labels == ["NumBasis 25", "NumBasis 49", "NumBasis 25・階段2倍細かく"]
    assert len(at.dataframe) >= 1
    verdicts = [m.value for m in at.success] + [m.value for m in at.warning]
    assert any('NumBasis 25 → 49' in v for v in verdicts) and any('階段近似を2倍' in v for v in verdicts)
    for r in cs['rows']:
        print(f"    {r['label']:24s} R0={np.round(r['R0'], 4)}  R+T={np.round(r['RT'], 4)}  {r['seconds']:.1f}s")

    # 「階段2倍」の行は n_slices=40 を直接解いた値と一致する
    import RCWA2D.rcwa_s4_array2d_backend as b2
    from RCWA2D.material_spec import resolve_nk_fn
    layers, fmap = build_layers([dict(cyl, n_slices=40)],
                                lattice=(np.array([532.0, 0.0]), np.array([0.0, 532.0])))
    direct = b2.calc_array_rcwa_s4(np.array([450.0, 650.0]), 0.0, 0.0, 0.532, 0.532, 25, layers, fmap,
                                   resolve_nk_fn(1.0), resolve_nk_fn('SiO2'))
    d = np.max(np.abs(direct['R0'] - np.array(cs['rows'][2]['R0'])))
    print(f"[2] 階段2倍の行 = n_slices 40 を直接解いた値（差 {d:.1e}）")
    assert d < 1e-9
    print("ALL OK")


if __name__ == "__main__":
    main()

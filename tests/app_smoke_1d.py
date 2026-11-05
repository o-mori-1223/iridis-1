"""iridis.py 1Dモード（列ベース構造）のスモーク＋回帰テスト。
streamlit.testing.v1.AppTest でブラウザなしにボタン操作を再現する。

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/app_smoke_1d.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                      # data/nk は cwd 相対で読むため
sys.path.insert(0, ROOT)

import numpy as np
from streamlit.testing.v1 import AppTest

# 2026-10-26 の分割前（iridis.py 単一ファイル版）で得た値。計算ロジックの回帰検出用。
REF_RP0_HEAD = [0.00319632, 0.00882731, 0.02225581]


def check(at, label):
    assert not at.exception, f"[{label}] 例外: {[str(e) for e in at.exception]}"
    print(f"[OK] {label}")


def main():
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=600)
    at.run()
    check(at, "初期表示 (1D)")

    for _ in range(2):
        at.button[0].click().run()          # ＋ 列を追加
    check(at, "列を2本追加")
    next(b for b in at.button if b.key == 'ladd_p_1').click().run()
    check(at, "列1にペア層を追加")

    at.button(key='calc_btn').click().run()
    check(at, "単一角度の計算")
    res = at.session_state['calc_cache']['result']
    rp0 = res['irp'][:, res['irp'].shape[1] // 2]
    assert not np.isnan(res['irp']).any()
    assert np.allclose(rp0[:3], REF_RP0_HEAD, atol=1e-8), f"Rp0 が基準値と違う: {rp0[:3]}"
    print(f"     Rp0[:3] = {rp0[:3]}（基準値と一致）")

    next(r for r in at.radio if r.label == 'Angle mode').set_value('Angle sweep (3D)').run()
    next(s for s in at.slider if s.label == 'Angle range [deg]').set_value((0.0, 40.0)).run()
    next(n for n in at.number_input if n.label == 'Angle step [deg]').set_value(20.0).run()
    at.button(key='calc_btn').click().run()
    check(at, "角度掃引（0/20/40°）＋スウォッチ(0/30/60°)の計算と表示")
    res = at.session_state['calc_cache']['result']
    assert res['Rs2d'].shape == (3, 81) and not np.isnan(res['Rs2d']).any()
    assert np.allclose(res['Rp2d'][0, :3], REF_RP0_HEAD, atol=1e-8), "角度掃引の0°が単一角度と一致しない"
    assert len(res['swatch']['Rp_single']) == 3
    print(f"     Rs2d.shape = {res['Rs2d'].shape}, スウォッチ3角度")
    print("ALL OK")


if __name__ == "__main__":
    main()

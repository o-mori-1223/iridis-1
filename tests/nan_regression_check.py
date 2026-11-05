"""2Dモードの NaN 回帰テスト（2026-10-26 にユーザー環境で発生した「基板段差の矩形領域で
データが NaN になり、Gsolver互換表示が消える」不具合）。

原因: 上げ領域が単位胞全体を覆う（例: ピッチ300x900nmに300x900nmの矩形）と、セル内に材質境界が
無く PolarizationDecomposition の法線ベクトル場が作れず、S4 の固有値問題が失敗して全点 NaN。
対策: rcwa_s4_array2d_backend.simplify_full_cell_regions（全面領域→背景材質、厳密に等価）と、
_solve_one_point の偏光分解なしへのフォールバック、color_display.fill_nan（表示側の補間）。

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/nan_regression_check.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

import RCWA2D.rcwa_s4_array2d_backend as b2
from RCWA2D.color_display import fill_nan
from RCWA2D.material_spec import resolve_nk_fn
from RCWA2D.structure_builder import regions_to_layers_2d

FILM = [{'type': 'pair', 'mat_a': 'TiO2', 'th_a': 51.0, 'mat_b': 'SiO2', 'th_b': 81.0, 'n_pairs': 7},
        {'type': 'simple', 'material': 'TiO2', 'thickness': 51.0}]
REGION = [{'shape': 'rect', 'cx_nm': 0.0, 'cy_nm': 0.0, 'hw_x_nm': 150, 'hw_y_nm': 450, 'raise_nm': 235.0}]


def main():
    fmap = {'TiO2': resolve_nk_fn('TiO2'), 'SiO2': resolve_nk_fn('SiO2')}
    vac, sub = resolve_nk_fn(1.0), resolve_nk_fn('SiO2')
    wl = np.array([400.0, 470.0, 540.0, 600.0, 700.0])
    layers = regions_to_layers_2d(REGION, FILM)

    # 1) 単位胞全体を覆う矩形（矩形格子）: NaN が出ず、等価な平坦構造と一致する
    flat = [{'thickness_nm': L['thickness_nm'],
             'bg_material': L['rects'][0]['material'] if L.get('rects') else L['bg_material'], 'disks': []}
            for L in layers]
    a = b2.calc_array_rcwa_s4_angle_sweep(wl, np.radians([0.0, 30.0]), 0.0, 0.3, 0.9, 49, layers, fmap, vac, sub)
    b = b2.calc_array_rcwa_s4_angle_sweep(wl, np.radians([0.0, 30.0]), 0.0, 0.3, 0.9, 49, flat, fmap, vac, sub)
    assert not np.isnan(a['R0']).any(), "全面矩形で NaN が出た"
    d = np.max(np.abs(a['R0'] - b['R0']))
    print(f"[OK] 全面矩形（矩形格子）: NaN なし、平坦構造との差 {d:.1e}")
    assert d < 1e-12

    # 2) 六方格子で全面を覆う場合（単純化の対象外）: 偏光分解なしへのフォールバックで有限値
    c = b2.calc_array_rcwa_s4(wl, 0.0, 0.0, 0.3, 0.3, 49, layers, fmap, vac, sub, gamma_deg=60.0)
    assert not np.isnan(c['R0']).any(), "六方格子・全面矩形で NaN が出た（フォールバックが効いていない）"
    rt = c['R_total'] + c['T_total']
    print(f"[OK] 全面矩形（六方格子、フォールバック）: NaN なし、R+T = {rt.min():.4f}〜{rt.max():.4f}")
    assert np.all(np.abs(rt - 1) < 0.02)

    # 3) 表示側: NaN が残っても補間して JSON に載せられる
    z = np.array([[0.1, np.nan, 0.3], [np.nan, np.nan, np.nan], [0.5, 0.6, np.nan]])
    filled, mask = fill_nan(z, [400, 500, 600])
    assert np.isfinite(filled).all() and mask.sum() == 5 and abs(filled[0, 1] - 0.2) < 1e-12
    print(f"[OK] 表示用 NaN 補間: {int(mask.sum())} 点を補間")
    print("ALL OK")


if __name__ == "__main__":
    main()

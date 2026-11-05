"""2Dモードの縦断面図（iridis_app/array2d_cross_section.py）の検証。S4計算なし。

1) 円盤層: y=0 の断面で、中心は円盤の材質、半径の外は背景、隣の周期像の円盤も正しく出る
2) 位置オフセットした円盤・六方格子でも、周期像を含めて正しく判定する
3) 入れ子の図形（レンズ状の凹凸の同心円・単位胞の角の島）は内側（小さい図形）が優先され、
   断面の材質が境界面の高さ（最も近い凹凸の中心からの距離で決まる）と一致する
4) 図が作れる（層構成・基板段差・レンズ・多層膜円盤）

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/cross_section_check.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

from iridis_app.array2d_cross_section import _materials_along_line, stack_cross_section_fig
from RCWA2D.curved_multilayer import build_layers
from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL, regions_to_layers_2d

SQ = (np.array([532.0, 0.0]), np.array([0.0, 532.0]))
HEX = (np.array([532.0, 0.0]), np.array([266.0, 532.0 * np.sqrt(3) / 2]))


def shifts(a1, a2, n=3):
    return [tuple(i * a1 + j * a2) for i in range(-n, n + 1) for j in range(-n, n + 1)]


def at(L, x, y, lat=SQ):
    return _materials_along_line(L, np.array([float(x)]), y, shifts(*lat))[0]


def main():
    disk = {'thickness_nm': 100.0, 'bg_material': ENV_MATERIAL,
            'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}
    assert at(disk, 0, 0) == 'TiO2' and at(disk, 149, 0) == 'TiO2'
    assert at(disk, 151, 0) == ENV_MATERIAL and at(disk, 266, 0) == ENV_MATERIAL
    assert at(disk, 532 - 140, 0) == 'TiO2'            # 隣の周期像
    assert at(disk, 0, 200) == ENV_MATERIAL and at(disk, 0, 140) == 'TiO2'
    print("[1] 円盤層: 中心・外側・隣の周期像・断面の位置 y の判定 OK")

    off = {'thickness_nm': 50.0, 'bg_material': 'SiO2',
           'disks': [{'material': 'TiO2', 'cx_nm': 100.0, 'cy_nm': 0.0, 'radius_nm': 100.0}]}
    assert at(off, 190, 0) == 'TiO2' and at(off, -10, 0) == 'SiO2'
    # 六方格子: (266, 460.7) の周期像の円盤が y=460 の断面に出る
    assert at(disk, 266, 460.0, HEX) == 'TiO2' and at(disk, 0, 460.0, HEX) == ENV_MATERIAL
    print("[2] 位置オフセット・六方格子の周期像 OK")

    recipe = [{'type': 'simple', 'material': 'TiO2', 'thickness': 51.0},
              {'type': 'pair', 'mat_a': 'TiO2', 'th_a': 51.0, 'mat_b': 'SiO2', 'th_b': 81.0, 'n_pairs': 3}]
    for lat in (SQ, HEX):
        layers, info = build_layers('spherical_cap', 150.0, 100.0, recipe, lattice=lat)
        names = [SUBST_MATERIAL] + info['materials'] + [ENV_MATERIAL]
        z_top = info['slices'][-1][1]
        xs = np.linspace(-266.0, 266.0, 533)
        sh = shifts(*lat)
        centers = np.array(sh)
        n_bad = n_all = 0
        z = z_top
        for L in layers:
            z_lo = z - L['thickness_nm']
            zm = 0.5 * (z + z_lo)
            mats = _materials_along_line(L, xs, 0.0, sh)
            d = np.hypot(xs[:, None] - centers[None, :, 0], 0.0 - centers[None, :, 1]).min(axis=1)
            zk = np.array([np.interp(d, info['x'], Zk) for Zk in info['Z']])   # [N+1, len(xs)]
            expect = [names[int((zk[:, i] <= zm).sum())] for i in range(len(xs))]
            # 階段近似の誤差（境界の近く）は除いて、境界面から離れた点だけ比べる
            far = np.abs(zk - zm).min(axis=0) > 0.6 * L['thickness_nm'] + 1.0
            n_bad += int(sum(m != e for m, e, f in zip(mats, expect, far) if f))
            n_all += int(far.sum())
            z = z_lo
        print(f"[3] レンズ状の凹凸（{'正方' if lat is SQ else '六方'}、島 {sum(len(L['polygons']) for L in layers)} 個）: "
              f"断面の材質が境界面の高さと一致しない点 {n_bad}/{n_all}")
        assert n_bad <= 0.002 * n_all

    from iridis_app.array2d_model import preview_color_map
    step_layers = regions_to_layers_2d([{'shape': 'rect', 'cx_nm': 0.0, 'cy_nm': 0.0, 'hw_x_nm': 100.0,
                                         'hw_y_nm': 100.0, 'raise_nm': 80.0}], recipe)
    _, info_d = build_layers('spherical_cap', 150.0, 100.0, recipe, concave=True, lattice=SQ, disk_radius_nm=200.0)
    layers_d, _ = build_layers('spherical_cap', 150.0, 100.0, recipe, concave=True, lattice=SQ, disk_radius_nm=200.0)
    for name, layers, kw in (('円盤層', [disk], {}), ('基板段差', step_layers, {}),
                             ('凹＋多層膜円盤', layers_d, dict(z_bottom_nm=info_d['slices'][0][0], lens_info=info_d))):
        fig = stack_cross_section_fig(layers, *SQ, 0.0, preview_color_map([], recipe), 'SiO2', n_periods=3, **kw)
        assert len(fig.data) >= 2
        print(f"[4] 図: {name} → トレース {len(fig.data)} 本")
    print("ALL OK")


if __name__ == "__main__":
    main()

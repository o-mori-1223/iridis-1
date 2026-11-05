"""RCWA2D.s4_parallel（プロセス並列）が直列計算（各 backend 関数）とビット単位で一致し、
停止要求が効くことを確認する。所要時間も表示する。

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/s4_parallel_check.py
"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np

import RCWA2D.rcwa_s4_array2d_backend as b2
import RCWA2D.rcwa_s4_backend as b1
from RCWA2D import s4_parallel
from RCWA2D.material_spec import resolve_nk_fn
from RCWA2D.structure_builder import columns_to_layers


def main():
    cols = [{'width': w, 'subst_raise': r,
             'film_layers': [{'type': 'pair', 'mat_a': 'TiO2', 'th_a': 77.6,
                              'mat_b': 'SiO2', 'th_b': 54.4, 'n_pairs': 7}]}
            for w, r in zip([900, 600, 300, 300, 600, 300], [0, 235, 0, 235, 0, 235])]
    layers = columns_to_layers(cols)
    fmap = {'TiO2': resolve_nk_fn('TiO2'), 'SiO2': resolve_nk_fn('SiO2')}
    sus = resolve_nk_fn('SUS')
    wl = np.linspace(380, 780, 41)
    angs = np.radians([0.0, 20.0, 40.0])
    print(f"workers = {s4_parallel.n_workers()}")

    t0 = time.time()
    P = s4_parallel.solve_1d(wl, angs, 3.0, 43, layers, fmap, 1.0, sus)
    tp = time.time() - t0
    t0 = time.time()
    S = [b1.calc_rcwa1d_s4(wl, a, 3.0, 43, layers, fmap, 1.0, sus) for a in angs]
    ts = time.time() - t0
    d = max(np.max(np.abs(p - s)) for pa, sa in zip(P, S) for p, s in zip(pa, sa))
    print(f"1D 6列DBR nOrder=43, 3角度x41波長: 並列 {tp:.2f}s / 直列 {ts:.2f}s  max|diff|={d:.1e}")
    assert d == 0.0

    disk = [{'thickness_nm': 100.0, 'bg_material': b2.ENV_MATERIAL,
             'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}]
    args = (0.0, 0.4, 0.4, 101, disk, {'TiO2': resolve_nk_fn('TiO2')}, resolve_nk_fn(1.0), resolve_nk_fn('SiO2'))
    th = np.radians([0.0, 15.0, 30.0, 45.0])
    t0 = time.time()
    P2 = s4_parallel.solve_2d(wl, th, *args, gamma_deg=60.0)
    tp = time.time() - t0
    t0 = time.time()
    S2 = b2.calc_array_rcwa_s4_angle_sweep(wl, th, *args, gamma_deg=60.0)
    ts = time.time() - t0
    d = max(np.max(np.abs(P2[k] - S2[k])) for k in ('R0', 'R_total', 'T0', 'T_total'))
    print(f"2D 六方格子円盤 NumBasis=101, 4角度x41波長: 並列 {tp:.2f}s / 直列 {ts:.2f}s  max|diff|={d:.1e}")
    assert d == 0.0 and P2['n_computed'] == th.size * wl.size

    ev = threading.Event()
    ev.set()
    P3 = s4_parallel.solve_2d(wl, th, *args, stop_event=ev)
    print(f"停止要求あり: n_computed={P3['n_computed']} / {th.size * wl.size}")
    assert P3['n_computed'] < th.size * wl.size
    print("ALL OK")


if __name__ == "__main__":
    main()

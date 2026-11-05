"""基板+円盤ケースでQ_scatter,upが物理的にあり得ない値になっていた件の
回帰検証スクリプト（2026-09-24にこのセッションで見つけた2つの独立なバグの
再発防止用。修正の経緯はMEEP3D/geometry.py・MEEP3D/simulate.pyのdocstring
参照）。

見つかった2つのバグ:
1. **flux boxの底面位置バグ（geometry.py）**: 修正前は `z_bottom = box_margin`
   が基板表面(z=0)からの固定オフセットで、円盤の高さ(height_um)を全く考慮
   していなかった。height_um > box_margin だと底面モニタが円盤内部を貫通し、
   3-run減算法（vacuum/bare_substrate/disk）の前提（モニタ面上の背景材質が
   run間で一致していること）が崩れていた。修正: 円盤を球のMie検証と同じ
   「全方向にbox_margin分の余白で完全に包む」閉じた箱にする
   （z_bottom = -box_margin、基板内部）。
2. **入射光量(i0)の符号バグ（simulate.py）**: 円盤パイプラインの光源は
   -z(下向き)伝搬だが、i0に使っていた'top'面はz法線でMEEPの符号規約は
   +zが正 — 伝搬方向と規約の正方向が逆なため、fluxA['top']は常に負の値に
   なる（sphere-in-vacuum検証は伝搬+xで、i0に使う'x-'面が規約の正方向と
   一致していたため、この非対称なアナロジーの見落としに気づかなかった）。
   これによりQ_scatter,upの符号が常に反転していた（無損失媒質なら常に0以上の
   はずが、常に負だった）。修正: i0として`-fluxA['top']`を渡す。

TEST1は円盤材質=基板材質（コントラストなし、幾何形状だけの摂動）、TEST2は
通常コントラスト、TEST4は円盤高さがbox_marginを明確に超えるケース(バグ1が
なければ機能しない設定)。無損失材質なので、3テストとも**Q_scatter,upは
常に0以上**になっているはずというのが唯一のサニティチェック
（球のMie検証diag_meep_sphere_mie.pyと同じ根拠）。

速度重視で低解像度・少ない周波数点数を使う（精密な収束確認は目的ではない、
符号・大きさの健全性チェックが目的）。

Run: conda activate mp && cd <repo> && python3 scripts/meep3d/diag_meep_disk_boxmargin.py
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/<分類>/ から見たリポジトリ直下
sys.path.insert(0, _ROOT)

import numpy as np
import meep as mp

from MEEP3D.geometry import build_cell
from MEEP3D.materials import dielectric_medium
from MEEP3D.simulate import _scattering_efficiency_from_fluxes


def _add_flux_all_faces(sim, cfg, fcen, df, nfreq):
    return {name: sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=c, size=s))
            for name, c, s in cfg.flux_box.faces}


def run_once(radius_nm, height_nm, box_margin_um, disk_medium, substrate_medium,
             resolution=20, n_freq=5, wl_min_nm=500.0, wl_max_nm=700.0, decay_by=1e-5,
             use_symmetry=True):
    r_um, h_um = radius_nm / 1000.0, height_nm / 1000.0
    fmin = 1.0 / (wl_max_nm / 1000.0)
    fmax = 1.0 / (wl_min_nm / 1000.0)
    fcen, df = 0.5 * (fmin + fmax), (fmax - fmin) * 1.5

    def _build(mode):
        cfg, cz = build_cell(mode, r_um, h_um, disk_medium, substrate_medium,
                              dpml=0.4, pad=0.4, box_margin=box_margin_um,
                              fcen=fcen, df=df, n_freq=n_freq, use_symmetry=use_symmetry)
        sim = mp.Simulation(cell_size=cfg.cell_size, boundary_layers=cfg.boundary_layers,
                             geometry=cfg.geometry, sources=cfg.sources, symmetries=cfg.symmetries,
                             resolution=resolution, geometry_center=mp.Vector3(0, 0, cz),
                             default_material=mp.Medium(index=1.0))
        return sim, cfg

    def _run(sim, cfg, load_from=None):
        fluxes = _add_flux_all_faces(sim, cfg, fcen, (fmax - fmin), n_freq)
        if load_from is not None:
            for name, obj in fluxes.items():
                sim.load_minus_flux_data(obj, load_from[name])
        pt = mp.Vector3(0, 0, cfg.flux_box.z_top)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt, decay_by))
        wl_nm = 1000.0 / np.array(mp.get_flux_freqs(fluxes['top']))
        vals = {name: np.array(mp.get_fluxes(obj)) for name, obj in fluxes.items()}
        data = {name: sim.get_flux_data(obj) for name, obj in fluxes.items()}
        return wl_nm, vals, data

    simA, cfgA = _build('vacuum')
    wl_nm, fluxA, _ = _run(simA, cfgA)
    simB, cfgB = _build('bare_substrate')
    _, fluxB, dataB = _run(simB, cfgB)
    simC, cfgC = _build('disk')
    _, fluxC, _ = _run(simC, cfgC, load_from=dataB)

    r_box = cfgC.flux_box.r_box
    print(f"  [height={height_nm:g}nm box_margin={box_margin_um*1000:g}nm r_box={r_box*1000:.1f}nm "
          f"z_bottom={cfgC.flux_box.z_bottom*1000:.1f}nm z_top={cfgC.flux_box.z_top*1000:.1f}nm "
          f"disk_top={h_um*1000:.1f}nm]")
    for name in ('top', 'bottom', 'x+', 'x-', 'y+', 'y-'):
        print(f"    face {name:6s}: {fluxC[name]}")
    # i0は-fluxA['top']（伝搬-z、'top'面のMEEP符号規約は+zが正なので反転が必要。
    # simulate.py run_scattering_spectrum()と同じ規約 -- 詳細はそちら参照）。
    q = _scattering_efficiency_from_fluxes(fluxC, r_box, r_um, -fluxA['top'])
    print(f"    wl(nm)={wl_nm}")
    print(f"    Q_scatter,up = {q}  (無損失材質なので全点0以上になっているはず)")
    return wl_nm, q


sio2 = dielectric_medium('SiO2')
tio2 = dielectric_medium('TiO2')

print('=== TEST 1: ゼロテスト(円盤材質=基板材質=SiO2), デフォルト寸法(height=100nm=box_margin) ===')
run_once(radius_nm=150.0, height_nm=100.0, box_margin_um=0.1, disk_medium=sio2, substrate_medium=sio2)

print()
print('=== TEST 2: 通常コントラスト(TiO2円盤/SiO2基板), デフォルト寸法(height=100nm=box_margin) ===')
run_once(radius_nm=150.0, height_nm=100.0, box_margin_um=0.1, disk_medium=tio2, substrate_medium=sio2)

print()
print('=== TEST 4: 通常コントラスト、円盤高さがbox_marginを明確に超える(height=300nm, box_margin=100nm) ===')
print('    バグ1(底面貫通)が残っていれば、ここだけ極端な負値(-4〜-6)になって表面化するはず')
run_once(radius_nm=150.0, height_nm=300.0, box_margin_um=0.1, disk_medium=tio2, substrate_medium=sio2)

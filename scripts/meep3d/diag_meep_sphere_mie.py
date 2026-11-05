"""検証用: MEEP公式のsphere-in-vacuum Mie散乱チュートリアルを再現し、
MEEP3D.simulate._scattering_efficiency_from_fluxes()（円盤パイプラインが
実際に使うのと同じ算術コード）が正しい散乱効率を返すかを確認する。

参照: https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/
（Mie scattering of a lossless dielectric sphere, n=2.0, r=1.0um）。
このスクリプトを実行して得られる Qsca(周波数) の振動パターン
（複数のMie共鳴ピーク、値のオーダーは概ね0〜4程度）を、公式チュートリアル
ページの掲載グラフと目視で突き合わせること。基板ありの円盤の結果は、
このチェックに合格するまで信用しないこと（HANDOVER.md参照）。

Run: conda activate mp && cd <repo> && python3 scripts/meep3d/diag_meep_sphere_mie.py
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/<分類>/ から見たリポジトリ直下
sys.path.insert(0, _ROOT)

import numpy as np
import meep as mp

from MEEP3D.simulate import _scattering_efficiency_from_fluxes

r = 1.0          # sphere radius (um)
n_sphere = 2.0

# 公式チュートリアルの数値をそのまま踏襲（WebFetchでページから直接確認済み）。
wvl_min = 2 * np.pi * r / 10.0
wvl_max = 2 * np.pi * r / 2.0
frq_min = 1.0 / wvl_max
frq_max = 1.0 / wvl_min
frq_cen = 0.5 * (frq_min + frq_max)
dfrq = frq_max - frq_min
nfrq = 100
resolution = 25
dpml = 0.5 * wvl_max
dair = 0.5 * wvl_max

s = 2.0 * (dpml + dair + r)
cell_size = mp.Vector3(s, s, s)
pml_layers = [mp.PML(dpml)]

symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

sources = [mp.Source(
    mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
    center=mp.Vector3(-0.5 * s + dpml),
    size=mp.Vector3(0, s, s),
    component=mp.Ez,
)]


def _make_sim(with_sphere):
    geometry = []
    if with_sphere:
        geometry = [mp.Sphere(material=mp.Medium(index=n_sphere), center=mp.Vector3(), radius=r)]
    return mp.Simulation(
        cell_size=cell_size, boundary_layers=pml_layers, sources=sources,
        symmetries=symmetries,
        geometry=geometry, resolution=resolution, default_material=mp.Medium(),
    )


def _add_box_flux(sim):
    faces = {}
    faces['x+'] = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2 * r, 2 * r)))
    faces['x-'] = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2 * r, 2 * r)))
    faces['y+'] = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2 * r, 0, 2 * r)))
    faces['y-'] = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r, 0, 2 * r)))
    # geometry.pyのface命名(top/bottom)に合わせ、z+をtop, z-をbottomとして扱う
    faces['top'] = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2 * r, 2 * r, 0)))
    faces['bottom'] = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2 * r, 2 * r, 0)))
    return faces


print('=== Run 1: 真空(球なし) -- 入射光量の基準値 ===')
sim1 = _make_sim(with_sphere=False)
flux1 = _add_box_flux(sim1)
sim1.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, mp.Vector3(), 1e-6))
freqs = np.array(mp.get_flux_freqs(flux1['x-']))
wl_um = 1.0 / freqs
i0 = np.array(mp.get_fluxes(flux1['x-']))
data1 = {name: sim1.get_flux_data(obj) for name, obj in flux1.items()}

print('=== Run 2: 球あり -- 散乱成分を抽出 ===')
sim2 = _make_sim(with_sphere=True)
flux2 = _add_box_flux(sim2)
for name, obj in flux2.items():
    sim2.load_minus_flux_data(obj, data1[name])
sim2.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, mp.Vector3(), 1e-6))
values2 = {name: np.array(mp.get_fluxes(obj)) for name, obj in flux2.items()}

q = _scattering_efficiency_from_fluxes(values2, r_box=r, r_geom=r, i0_flux=i0)

print()
print(f"{'wl(um)':>8} {'freq':>8} {'Qsca':>10}")
for i in range(0, nfrq, max(1, nfrq // 20)):
    print(f"{wl_um[i]:>8.4f} {freqs[i]:>8.4f} {q[i]:>10.4f}")

print()
print('公式チュートリアルの掲載グラフ(Qsca vs frequency, 複数の共鳴ピーク、')
print('値のオーダーは概ね0〜4程度)と目視で比較すること。')

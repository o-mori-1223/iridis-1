"""Example: broadband 2D dispersive-FDTD structural-color simulation.

Reproduces the kind of structure sketched by the user: a laterally periodic surface
relief on a metal (SUS) substrate, where each period is a handful of side-by-side
"columns" raised to different heights and each carrying the same conformal multilayer
thin-film stack -- i.e. exactly the "column-periodic" structure format already used by
the RCWA side of this project (see RCWA2D/structure_builder.py / R8_koumyou_v3.py). Edit the
`columns` list below to match your actual target geometry (widths, step heights,
materials, film thicknesses); everything else in this script stays the same.

Run:  .venv/Scripts/python.exe fdtd_demo.py
"""
import numpy as np

from fdtd2d.simulate import run_spectrum
from fdtd2d.color import spectrum_to_srgb

# ---------------------------------------------------------------------------
# Structure: one period = 4 columns forming a raised terrace (staircase up, then
# down), each carrying the same TiO2/SiO2 pair stack conformally on top, on a SUS
# substrate. This mirrors the "column-periodic" film format from RCWA2D/structure_builder.py:
# each column is {width, subst_raise, film_layers}.
# ---------------------------------------------------------------------------
PAIR_STACK = [{'type': 'pair', 'mat_a': 'TiO2', 'th_a': 60.0,
               'mat_b': 'SiO2', 'th_b': 90.0, 'n_pairs': 4}]

columns = [
    {'width': 300.0, 'subst_raise': 0.0,   'film_layers': PAIR_STACK},
    {'width': 300.0, 'subst_raise': 150.0, 'film_layers': PAIR_STACK},
    {'width': 300.0, 'subst_raise': 300.0, 'film_layers': PAIR_STACK},
    {'width': 300.0, 'subst_raise': 150.0, 'film_layers': PAIR_STACK},
]

nk_subst_name = 'SUS'
n_env = 1.0            # ambient (air)
wl_min_nm, wl_max_nm = 400.0, 800.0
n_wl = 41               # spectral points
dx_nm = 6.0             # grid cell size (isotropic, x=z); finer = more accurate & slower
n_poles = 5             # dispersion poles fitted per material (see fdtd2d/materials.py)

if __name__ == '__main__':
    result = run_spectrum(columns, nk_subst_name, n_env, wl_min_nm, wl_max_nm,
                           n_wl=n_wl, dx_nm=dx_nm, n_poles=n_poles, verbose=True)

    hex_color, RGB, XYZ = spectrum_to_srgb(result.wl_nm, result.R)

    print()
    print(f'pitch = {result.grid.pitch_nm:.1f} nm   grid = {result.grid.nx} x {result.grid.nz} cells')
    print(f'{"wl(nm)":>8} {"R":>8} {"T":>8} {"A":>8}')
    for wl, R, T, A in zip(result.wl_nm, result.R, result.T, result.A):
        print(f'{wl:8.1f} {R:8.4f} {T:8.4f} {A:8.4f}')
    print()
    print(f'sRGB color of the reflected spectrum: {hex_color}  (RGB={RGB}, XYZ={XYZ})')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(result.wl_nm, result.R, label='R (reflectance)')
        ax.plot(result.wl_nm, result.T, label='T (transmittance)')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('R / T')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f'FDTD structural color spectrum  (color = {hex_color})')
        fig.patch.set_facecolor(hex_color)
        out_path = 'fdtd_demo_spectrum.png'
        fig.savefig(out_path, dpi=150)
        print(f'saved plot to {out_path}')
    except Exception as exc:
        print(f'(skipping plot: matplotlib unavailable/broken in this venv -- {exc})')

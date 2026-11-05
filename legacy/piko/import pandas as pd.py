import pandas as pd
import numpy as np
import colour
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

# ============================================================
# 設定
# ============================================================
INPUT_CSV     = "input/trapezoid_470.csv"
OUTPUT_CSV    = "result/csv/trapezoid_470_selected.csv"
OUTPUT_PNG    = "result/png/trapezoid_470_color.png"
TARGET_ANGLES = [0, 30, 60]            # 反射率・色を詳細表示する入射角 [deg]
VA_ANGLES     = list(range(0, 81, 10)) # 視野角カラーブロック用（0°〜80°, 10°刻み）

# ============================================================
# 1. COMSOL CSV 読み込み
# ============================================================
df = pd.read_csv(INPUT_CSV, skiprows=5, header=None,
                 names=["lamda_nm", "theta_deg", "freq_THz", "R0"])
df["lamda_nm"]  = pd.to_numeric(df["lamda_nm"],  errors="coerce")
df["theta_deg"] = pd.to_numeric(df["theta_deg"], errors="coerce")
df["freq_THz"]  = pd.to_numeric(df["freq_THz"],  errors="coerce")
df["R0"]        = pd.to_numeric(df["R0"],         errors="coerce")
df = df.dropna(subset=["lamda_nm", "theta_deg", "R0"])

# ============================================================
# 2. 行列変換（行：角度，列：波長）＋ 波長の丸め
# ============================================================
matrix = df.pivot(index="theta_deg", columns="lamda_nm", values="R0")
matrix = matrix.sort_index().sort_index(axis=1)
matrix.columns = np.round(matrix.columns.astype(float), 0).astype(int)

# ============================================================
# 3. 対象角度を最近傍で選択 → CSV 保存
# ============================================================
available_angles = matrix.index.astype(float).values

selected_angles = [
    available_angles[np.argmin(np.abs(available_angles - ta))]
    for ta in TARGET_ANGLES
]
labels = [f"θ={ta}°" for ta in TARGET_ANGLES]

extracted = matrix.loc[selected_angles].copy()
extracted.index = labels
extracted.to_csv(OUTPUT_CSV)
print(f"抽出済みCSVを保存しました: {OUTPUT_CSV}")

# ============================================================
# 4. 光源・等色関数の取得
# ============================================================
illuminant  = colour.SDS_ILLUMINANTS['D65']
cmfs        = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
wp_D65_xy   = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
wavelengths = matrix.columns.astype(float).values

# ============================================================
# 5. 色計算ユーティリティ
# ============================================================
def reflectance_to_sRGB(ref_values):
    ref = ref_values.astype(float)
    if np.nanmax(ref) > 1.0:
        ref = ref / 100.0
    ref = np.clip(np.nan_to_num(ref, nan=0.0), 0.0, 1.0)
    sd   = colour.SpectralDistribution(ref, wavelengths)
    XYZ  = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant)
    sRGB = colour.XYZ_to_sRGB(XYZ / 100.0, illuminant=wp_D65_xy, apply_cctf_encoding=True)
    return np.clip(sRGB, 0, 1), XYZ

def normalize_ref(ref_values):
    ref = ref_values.astype(float)
    if np.nanmax(ref) > 1.0:
        ref = ref / 100.0
    return np.clip(np.nan_to_num(ref, nan=0.0), 0.0, 1.0)

# ============================================================
# 6. TARGET_ANGLES の色計算
# ============================================================
colors, XYZ_list, all_ref = [], [], []
for label in labels:
    ref_raw = extracted.loc[label].values
    sRGB, XYZ = reflectance_to_sRGB(ref_raw)
    colors.append(sRGB)
    XYZ_list.append(XYZ)
    all_ref.append(normalize_ref(ref_raw))

for label, XYZ, rgb in zip(labels, XYZ_list, colors):
    print(f"{label}  XYZ={XYZ}  sRGB={rgb}  RGB(0-255)={np.round(rgb*255).astype(int)}")

# ============================================================
# 7. VA_ANGLES の色計算
# ============================================================
va_selected_angles = [
    available_angles[np.argmin(np.abs(available_angles - ta))]
    for ta in VA_ANGLES
]
va_colors = []
for theta in va_selected_angles:
    sRGB, _ = reflectance_to_sRGB(matrix.loc[theta].values)
    va_colors.append(sRGB)

# ============================================================
# 8. 2D 可視化（スペクトル + 色パッチ + 視野角カラーブロック）→ PNG 保存
# ============================================================
def wavelength_to_rgb(wl):
    xyz = colour.wavelength_to_XYZ(wl)
    rgb = colour.XYZ_to_sRGB(xyz, apply_cctf_encoding=True)
    return np.clip(rgb, 0, 1)

spectral_colors = [wavelength_to_rgb(w) for w in wavelengths]
step    = int(np.median(np.diff(wavelengths)))
n_ang   = len(labels)
n_va    = len(VA_ANGLES)

fig = plt.figure(figsize=(12, 3 * n_ang + 3))
gs  = GridSpec(n_ang + 1, 2, figure=fig,
               height_ratios=[3] * n_ang + [1.5],
               hspace=0.45, wspace=0.3)

for i, label in enumerate(labels):
    ref = all_ref[i]

    ax_spec = fig.add_subplot(gs[i, 0])
    ax_spec.bar(wavelengths, ref, width=step * 0.9,
                color=spectral_colors, edgecolor='none')
    ax_spec.plot(wavelengths, ref, color='black', linewidth=1, alpha=0.4)
    ax_spec.set_title(f"Reflectance Spectrum ({label})")
    ax_spec.set_xlim(370, 790)
    ax_spec.set_ylim(0, 1.05)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Reflectance")

    ax_patch = fig.add_subplot(gs[i, 1])
    ax_patch.imshow([[colors[i]]])
    rgb_int    = np.round(colors[i] * 255).astype(int)
    brightness = np.mean(colors[i])
    ax_patch.text(0, 0, f"RGB: {rgb_int}",
                  color='white' if brightness < 0.5 else 'black',
                  ha='center', va='center', fontsize=13, fontweight='bold')
    ax_patch.set_title(f"Calculated Color ({label})")
    ax_patch.axis('off')

ax_va = fig.add_subplot(gs[n_ang, :])
ax_va.imshow(np.array(va_colors).reshape(1, n_va, 3), aspect='equal')
for j, angle in enumerate(VA_ANGLES):
    ax_va.text(j, -0.45, f"{angle}°",
               color='black', ha='center', va='bottom',
               fontsize=10, fontweight='bold',
               transform=ax_va.get_xaxis_transform())
ax_va.set_xticks([])
ax_va.set_yticks([])
ax_va.set_title("Viewing Angle Dependence: Color vs. Incident Angle (0° – 80°)",
                fontsize=11, pad=8)

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"グラフを {OUTPUT_PNG} に保存しました")

# ============================================================
# 9. 3D 曲面グラフ（インタラクティブ表示）
# ============================================================
available_wl = matrix.columns.astype(int).values
selected_wl  = sorted(set(
    available_wl[np.argmin(np.abs(available_wl - wl))]
    for wl in range(380, 781, 10)
))

angles_3d = matrix.index.astype(float).values
dense_wl  = np.linspace(min(selected_wl), max(selected_wl), 60)
R_dense   = np.array([
    np.interp(dense_wl, selected_wl,
              matrix[selected_wl].loc[theta].values.astype(float))
    for theta in angles_3d
])

W, A = np.meshgrid(dense_wl, angles_3d)

wavelength_cmap = LinearSegmentedColormap.from_list(
    "wavelength_cmap",
    [(0.00, "purple"), (0.23, "blue"), (0.47, "green"),
     (0.67, "orange"), (1.00, "red")]
)
facecolors = wavelength_cmap((W - W.min()) / (W.max() - W.min()))

fig3d = plt.figure(figsize=(9, 6))
ax3d  = fig3d.add_subplot(111, projection="3d")

ax3d.plot_surface(
    W, A, R_dense,
    facecolors=facecolors,
    rstride=1, cstride=1,
    linewidth=0.3, edgecolor='k',
    alpha=1.0, antialiased=False
)

for wl in selected_wl:
    color = plt.cm.rainbow(
        (wl - min(selected_wl)) / (max(selected_wl) - min(selected_wl))
    )
    ax3d.plot([wl] * len(angles_3d), angles_3d,
              matrix[wl].values.astype(float),
              color=color, linewidth=1.0)

ax3d.set_xlabel("Wavelength λ [nm]")
ax3d.set_ylabel("Angle [deg]")
ax3d.set_zlabel("Reflectivity")
ax3d.set_title(INPUT_CSV.split("/")[-1].replace(".csv", ""))
ax3d.set_xlim(min(selected_wl), max(selected_wl))
ax3d.set_ylim(min(angles_3d), max(angles_3d))
ax3d.set_zlim(0, 1)
ax3d.view_init(elev=25, azim=45)

plt.subplots_adjust(bottom=0.18)

elev = [25]
azim = [45]
STEP = 10

def update(_=None):
    ax3d.view_init(elev=elev[0], azim=azim[0])
    fig3d.canvas.draw_idle()

btn_specs = [
    ([0.08, 0.04, 0.12, 0.05], "Az ←", lambda _: (azim.__setitem__(0, azim[0] - STEP), update())),
    ([0.22, 0.04, 0.12, 0.05], "Az →", lambda _: (azim.__setitem__(0, azim[0] + STEP), update())),
    ([0.44, 0.04, 0.12, 0.05], "El ↑", lambda _: (elev.__setitem__(0, min(elev[0] + STEP,  90)), update())),
    ([0.58, 0.04, 0.12, 0.05], "El ↓", lambda _: (elev.__setitem__(0, max(elev[0] - STEP, -90)), update())),
]
btns = []
for rect, lbl, cb in btn_specs:
    ax_b = fig3d.add_axes(rect)
    b = Button(ax_b, lbl)
    b.on_clicked(cb)
    btns.append(b)

plt.show()

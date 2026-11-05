import pandas as pd
import numpy as np
import colour
import matplotlib.pyplot as plt

# --- 1. CSV読み込み ---
df = pd.read_csv("maltlayer_ver2.1.csv", header=None)
angles = [0, 30, 60]

# 2) 波長軸
step = 10
n = len(df)
wavelengths = np.arange(380, 380 + step*n, step) 

# --- 3. 光源・等色関数の取得 ---
illuminant = colour.SDS_ILLUMINANTS['D65']
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
wp_D65_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# --- 4. 各列（角度）ごとに色を計算 ---
colors, XYZ_list = [], []
for i, angle in enumerate(angles):
    reflectance_values = df.iloc[:, i].to_numpy(dtype=float)
    # 値の健全化
    if np.nanmax(reflectance_values) > 1.0:
        reflectance_values = reflectance_values / 100.0
    reflectance_values = np.nan_to_num(reflectance_values, nan=0.0)
    reflectance_values = np.clip(reflectance_values, 0.0, 1.0)

    # SD 作成（長さ一致が重要）
    sd = colour.SpectralDistribution(reflectance_values, wavelengths)

    XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant)
    sRGB = colour.XYZ_to_sRGB(XYZ/100.0, illuminant=wp_D65_xy, apply_cctf_encoding=True)
    sRGB = np.clip(sRGB, 0, 1)

    colors.append(sRGB)
    XYZ_list.append(XYZ)

# --- 5. 結果出力 ---
for a, XYZ, rgb in zip(angles, XYZ_list, colors):
    print(f"θ={a}°  XYZ={XYZ} → sRGB={rgb} → RGB(0-255)={np.round(rgb*255).astype(int)}")

# --- 6. 色パッチで可視化 ---
fig, ax = plt.subplots(figsize=(6,3))
ax.imshow([colors])
ax.set_xticks(np.arange(len(angles)))
ax.set_xticklabels([f"θ={a}°" for a in angles], fontsize=11)
ax.tick_params(axis='x', pad=8)
ax.set_yticks([])
ax.set_title("Reflected Color by Angle", pad=12, fontsize=13)
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.88, bottom=0.2)
plt.show()
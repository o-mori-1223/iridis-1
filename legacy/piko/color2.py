import pandas as pd
import numpy as np
import colour
import matplotlib.pyplot as plt

# --- 1. CSV読み込み ---
df = pd.read_csv("maltlayer_ver2.1.csv", header=None)
angles = [0, 30, 60]  # 列の角度対応
print(df.head())

# --- 2. 波長範囲（例: 380〜780nm、10nm刻み）---
wavelengths = np.arange(380, 380 + 5 * len(df), 10)

# --- 3. 光源・等色関数の取得 ---
illuminant = colour.SDS_ILLUMINANTS['D65']
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
wp_D65_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# --- 4. 各列（角度）ごとに色を計算 ---
colors = []
XYZ_list = []

for i, angle in enumerate(angles):
    reflectance_values = df.iloc[:, i].to_numpy()

    # SpectralDistribution オブジェクト作成
    sd = colour.SpectralDistribution(reflectance_values, wavelengths)

    # XYZ計算
    XYZ = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant)
    XYZ_list.append(XYZ)

    # sRGB変換
    sRGB = colour.XYZ_to_sRGB(XYZ / 100, illuminant=wp_D65_xy, apply_cctf_encoding=True)
    sRGB = np.clip(sRGB, 0, 1)
    colors.append(sRGB)

# --- 5. 結果出力 ---
for a, XYZ, rgb in zip(angles, XYZ_list, colors):
    print(f"θ={a}°  XYZ={XYZ} → sRGB={rgb} → RGB(0-255)={np.round(rgb*255).astype(int)}")

# --- 6. 色パッチで可視化 ---
fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow([colors])  # 横に並べて1行表示
ax.set_xticks(np.arange(len(angles)))
ax.set_xticklabels([f"θ={a}°" for a in angles], fontsize=11)
ax.tick_params(axis='x', pad=8)
ax.set_yticks([])
ax.set_title("Reflected Color by Angle", pad=10, fontsize=14)
plt.show()
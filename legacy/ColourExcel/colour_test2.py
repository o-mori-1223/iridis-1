import numpy as np
import pandas as pd
import colour
from PIL import Image  # <-- ★ 1. ここに追加

#XYZ ;sRGB
# --- 1. 分光データの準備 --
# ( ... 省略 ... )
wavelengths = np.arange(380, 781, 5)
reflectance_values = np.zeros_like(wavelengths, dtype=float)
reflectance_values[(wavelengths >= 400) & (wavelengths <= 500)] = 0.70
reflectance_values[reflectance_values == 0.0] = 0.05
sample_reflectance = colour.SpectralDistribution(
    data=reflectance_values,
    domain=wavelengths,
    name='Sample Reflectance'
 )
 
# --- 2. 標準光源と等色関数の取得 --
# ( ... 省略 ... )
illuminant = colour.SDS_ILLUMINANTS['D65']
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
whitepoint_D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

# --- 3. 反射率からXYZへの変換 --
# ( ... 省略 ... )
XYZ = colour.sd_to_XYZ(
    sample_reflectance,
    cmfs=cmfs,
    illuminant=illuminant
 )
 
# --- 4. XYZからsRGBへの変換 --
# ( ... 省略 ... )
XYZ_normalised = XYZ / 100.0
sRGB_float = colour.XYZ_to_sRGB(
    XYZ_normalised,
    illuminant=whitepoint_D65,
    apply_cctf_encoding=True
)

# --- 5. sRGB値のクリッピングと16進数コードへの変換 (app.py のロジック) ---
sRGB_255_scaled = sRGB_float * 255.0
sRGB_clipped_int = np.clip(np.round(sRGB_255_scaled), 0, 255).astype(int)
R_255 = sRGB_clipped_int[0]
G_255 = sRGB_clipped_int[1]
B_255 = sRGB_clipped_int[2]
hex_code = f"#{R_255:02x}{G_255:02x}{B_255:02x}"


# --- 6. 結果の表示 --
print("--- Colour Science 変換結果 ---")
# print(f"分光反射率データ:\n{sample_reflectance}\n") # (表示が長いのでコメントアウトしてもOK)
print(f"計算されたXYZ値 (Y=輝度):\n{XYZ}")
print("-" * 30)
print(f"sRGB値 (0.0-1.0, クリップ前):\n{sRGB_float}")
print(f"sRGB値 (0-255, クリップ後):\n(R: {R_255}, G: {G_255}, B: {B_255})")
print(f"16進数カラーコード (可視化用):\n{hex_code}")


# --- ★ 7. 色の自動可視化 (ここから追加) ---

# (R, G, B) のタプルを作成
color_rgb_tuple = (R_255, G_255, B_255)

# 200x200ピクセルの単色画像を作成
img = Image.new('RGB', (200, 200), color=color_rgb_tuple)

# 画像をファイルとして保存
output_filename = 'calculated_color.png'
img.save(output_filename)

print("-" * 30)
print(f"色を {output_filename} として保存しました。")

# (オプション) 画像をデフォルトのビューアで開く
try:
    img.show()
    print("画像ビューアで表示します。")
except Exception as e:
    print(f"自動表示に失敗しました: {e}")
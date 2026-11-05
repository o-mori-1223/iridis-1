import numpy as np
import pandas as pd
import colour

#XYZ ;sRGB
# --- 1. 分光データの準備 --
# ユーザーの分光反射率データ (例として、5nm刻みのデータを使用)
 # 380nmから780nmまでの波長
wavelengths = np.arange(380, 781, 5)
 # デモ用: 架空の「青っぽい」反射率データ R(lambda) を生成
# 青い波長(400-500nm)で高い反射率、その他の波長で低い反射率を設定
reflectance_values = np.zeros_like(wavelengths, dtype=float)
 # 青の波長帯 (400nm - 500nm) の反射率を 70% に設定
reflectance_values[(wavelengths >= 400) & (wavelengths <= 500)] = 0.70
 # 赤・緑の波長帯の反射率を 5% に設定
reflectance_values[reflectance_values == 0.0] = 0.05
 # colour-scienceが扱うための分光分布オブジェクトを作成
 # ここで波長と反射率の値を結びつけています
sample_reflectance = colour.SpectralDistribution(
    data=reflectance_values,
    domain=wavelengths,
    name='Sample Reflectance'
 )
 # --- 2. 標準光源と等色関数の取得 --
# CIE D65 標準光源の分光分布データを取得
illuminant = colour.SDS_ILLUMINANTS['D65']
 # CIE 1931 2度視野の等色関数（XYZ変換の核となるデータ）を取得
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
 # D65白色点のXYZ座標を取得 (CIE 1931 2度視野)
whitepoint_D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

 # --- 3. 反射率からXYZへの変換 --
# colour.sd_to_XYZ() 関数を使用して、XYZ三刺激値を計算
XYZ = colour.sd_to_XYZ(
    sample_reflectance,
    cmfs=cmfs,
    illuminant=illuminant
 )
 
# --- 4. XYZからsRGBへの変換 --
# XYZを 0-1.0 のスケールに正規化 (Y=100.0 -> Y=1.0)
XYZ_normalised = XYZ / 100.0
# colour.XYZ_to_sRGB() 関数を使用して、sRGB値 (0.0-1.0) に変換
sRGB_float = colour.XYZ_to_sRGB(
    XYZ_normalised,
    illuminant=whitepoint_D65, # D65白色点
    apply_cctf_encoding=True # ガンマ補正を適用して非線形sRGBにする
)

# --- 5. sRGB値のクリッピングと16進数コードへの変換 (app.py のロジック) ---

# sRGB_float (0.0-1.0) を (0-255) のスケールに変換
sRGB_255_scaled = sRGB_float * 255.0

# 四捨五入して整数にし、0-255の範囲にクリッピング
sRGB_clipped_int = np.clip(np.round(sRGB_255_scaled), 0, 255).astype(int)

R_255 = sRGB_clipped_int[0]
G_255 = sRGB_clipped_int[1]
B_255 = sRGB_clipped_int[2]

# 16進数カラーコードを生成 (#RRGGBB 形式)
# :02x は「2桁の16進数で、足りない桁は0で埋める」という意味
hex_code = f"#{R_255:02x}{G_255:02x}{B_255:02x}"


# --- 6. 結果の表示 --
print("--- Colour Science 変換結果 ---")
print(f"分光反射率データ:\n{sample_reflectance}\n")
print(f"計算されたXYZ値 (Y=輝度):\n{XYZ}")
print("-" * 30)
print(f"sRGB値 (0.0-1.0, クリップ前):\n{sRGB_float}")
print(f"sRGB値 (0-255, クリップ後):\n(R: {R_255}, G: {G_255}, B: {B_255})")
print(f"16進数カラーコード (可視化用):\n{hex_code}")
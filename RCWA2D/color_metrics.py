"""反射スペクトル → 色（XYZ・CIELAB・sRGB）と、色の差 ΔE₀₀（CIEDE2000）。

色の計算は画面の表示（RCWA2D/color_display_template.html、iridis_app/column_app._sd_to_hex）と同じ
CIE 1931 2° 標準観測者・D65 光源。スペクトルは 380〜780 nm・5 nm 刻み（CIE_WL）で受け取る
（他の波長で計算したものは to_cie_grid で線形補間する）。多数の候補をまとめて扱えるよう、すべて
最後の軸が波長（または XYZ・Lab の 3 成分）の配列を受け取る。
"""
import colour
import numpy as np

CIE_WL = np.arange(380.0, 781.0, 5.0)

_CMFS = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
_D65 = colour.SDS_ILLUMINANTS['D65']
_W = np.asarray(_CMFS[CIE_WL], float) * np.asarray(_D65[CIE_WL], float)[:, None]   # [波長, XYZ]
_K = 100.0 / _W[:, 1].sum()
# Lab の白色点。sRGB の白（D65）と揃えるため、色度座標から求めた標準の値を使う
WHITE_XYZ = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']) * 100.0


def to_cie_grid(wl_nm, values):
    """任意の波長のスペクトル（最後の軸が波長）→ CIE_WL 上の値（範囲外は端の値）。"""
    values = np.asarray(values, float)
    flat = values.reshape(-1, values.shape[-1])
    out = np.array([np.interp(CIE_WL, wl_nm, row) for row in flat])
    return out.reshape(values.shape[:-1] + (CIE_WL.size,))


def spectrum_to_xyz(R):
    """CIE_WL 上の反射率 [..., 81] → XYZ [..., 3]（完全な白 R=1 で Y = 100）。"""
    return _K * np.asarray(R, float) @ _W


def xyz_to_lab(xyz):
    t = np.asarray(xyz, float) / WHITE_XYZ
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def spectrum_to_lab(R):
    return xyz_to_lab(spectrum_to_xyz(R))


def xyz_to_hex(xyz):
    rgb = np.clip(colour.XYZ_to_sRGB(np.asarray(xyz, float) / 100.0), 0.0, 1.0)
    return '#{:02x}{:02x}{:02x}'.format(*(int(round(v * 255)) for v in rgb))


def hex_to_rgb01(hex_str):
    h = hex_str.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def hex_to_lab(hex_str):
    xyz = colour.sRGB_to_XYZ(hex_to_rgb01(hex_str)) * 100.0
    return xyz_to_lab(xyz)


def example_spectrum(hex_str, smooth=1.0, small=0.01):
    """色 → その色になる反射スペクトルの例（CIE_WL 上、0 ≤ R ≤ 1）と、そのスペクトルの色・目標との ΔE₀₀。

    同じ色になるスペクトルは無数にあるので、なめらか（2 階差分の二乗和が小さい）で、反射率が全体に小さいもの
    （目の感度の高い波長に反射を置く。青なら 450 nm 付近、赤なら 610 nm 付近に山ができる）を 1 つ選ぶ。
    反射率 ≤ 1 では出せない明るく鮮やかな色のときは、出せる範囲で一番近いものになり、ΔE₀₀ が大きくなる。"""
    from scipy.optimize import lsq_linear
    xyz = colour.sRGB_to_XYZ(hex_to_rgb01(hex_str)) * 100.0
    n = CIE_WL.size
    D2 = np.diff(np.eye(n), 2, axis=0)
    # 色（XYZ）を合わせる行に大きな重みをかけ、ほかは弱く効かせる。暗い色ほど XYZ の差が小さくても目立つので、
    # Y で割って相対的な差にする
    w = 30.0 / max(xyz[1], 1.0) ** 0.5
    A = np.vstack([w * _K * _W.T, np.sqrt(smooth) * D2, np.sqrt(small) * np.eye(n)])
    b = np.concatenate([w * xyz, np.zeros(2 * n - 2)])
    R = lsq_linear(A, b, bounds=(0.0, 1.0), method='bvls').x
    got = spectrum_to_xyz(R)
    return R, xyz_to_hex(got), float(delta_e2000(xyz_to_lab(got), xyz_to_lab(xyz)))


def delta_e2000(lab1, lab2, ignore_lightness=False):
    """CIEDE2000（kL = kC = kH = 1）。ignore_lightness=True なら明度の差 ΔL' の項を除く
    （色相と彩度だけを比べる）。配列は最後の軸が L, a, b で、ブロードキャストできる形。"""
    lab1, lab2 = np.asarray(lab1, float), np.asarray(lab2, float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm = 0.5 * (L1 + L2)
    Cpm = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hpm = np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2)
    hpm = np.where(C1p * C2p == 0, hsum, hpm)
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    tL = 0.0 if ignore_lightness else dLp / Sl
    tC, tH = dCp / Sc, dHp / Sh
    return np.sqrt(tL ** 2 + tC ** 2 + tH ** 2 + Rt * tC * tH)

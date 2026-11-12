"""2Dモード「周期領域と円柱の切り抜き」の設定欄。

1×1・3×3・任意の Nx×Ny 周期をひとまとまり（周期領域）として計算し、その内接円（または指定の半径）の
円柱で構造を切り抜く。「内側を残す」なら、円柱の直径を直接指定することもできる（中の格子は円柱を埋めるのに
要る数だけ自動で並べる）。計算は RCWA2D/cylinder_cut.py（S4 に渡す層を並べてから切り抜く）。
有効なら設定（spec）を返し、無効なら None。spec はそのまま解析記録の params にも入る。
"""
import math

import numpy as np
import streamlit as st

from RCWA2D.cylinder_cut import (_unit_pitch, cut_radius_limit, inscribed_radius, region_lattice, supercell_vectors,
                                 tiles_for_diameter)
from RCWA2D.rcwa_s4_array2d_backend import ENV_MATERIAL

from .array2d_editor import material_selectbox

PRESETS = {"1×1": (1, 1), "3×3": (3, 3), "任意（Nx×Ny）": None}
MODES = {"keep": "内側を残す（外側を取り除く）", "hole": "内側を抜く（円柱の穴）"}
# 円柱（周期領域）の並べ方。値は RCWA2D/cylinder_cut.ARRANGEMENTS（'tile' は旧来の記録の読み込み用で画面には出さない）
ARRANGEMENT_LABELS = {"square": "正方配置", "hex": "六方最密配置"}
# 円柱の大きさの決め方（「内側を残す」のときだけ選べる）
SIZE_MODES = {"inscribed": "Nx×Ny 周期の内接円", "diameter": "直径を指定"}
NUMBASIS_MAX = 801   # サイドバーの NumBasis 入力欄の上限と同じ


def recommended_numbasis(base, factor):
    """周期の面積を factor 倍（Nx×Ny など）にしたとき、1×1 で base と同じ細かさになる NumBasis（奇数、上限あり）。"""
    nb = int(math.ceil(int(base) * float(factor) - 1e-9))
    nb += (nb + 1) % 2
    return min(nb, NUMBASIS_MAX), nb


def _set_numbasis(value):
    st.session_state["norder"] = int(value)   # on_click なので、サイドバーの入力欄より前に書き換えられる


def render_supercell_editor(nk_namelist, a1, a2, norder_now):
    enabled = st.checkbox(
        "複数の周期をひとまとまりにして計算し、円柱で切り抜く", value=False, key="sc_enabled",
        help="1×1・3×3 などの周期領域（スーパーセル）を新しい周期として計算する。円柱で切り抜くと、"
             "円柱の中の構造だけを残す（または円柱の穴をあける）。いまの層構成・基板段差・レンズ状の凹凸の"
             "すべてに効く。")
    if not enabled:
        return None

    cut = st.checkbox("円柱で切り抜く", value=True, key="sc_cut",
                      help="外すと、周期を並べるだけ（元の周期構造と同じ。NumBasis の確認用など）。")
    mode, size_mode, out = None, "inscribed", None
    if cut:
        m1, m2 = st.columns([1.3, 1])
        with m1:
            mode = st.radio("残し方", list(MODES), format_func=MODES.get, horizontal=True, key="sc_mode")
        with m2:
            out = material_selectbox(nk_namelist, "外側（穴）の材質", ENV_MATERIAL, "sc_out", allow_env_same=True,
                                     help="「内側を残す」なら円柱の外側、「内側を抜く」なら穴の中の材質。")
        if mode == "keep":
            size_mode = st.radio("円柱の大きさ", list(SIZE_MODES), format_func=SIZE_MODES.get, horizontal=True,
                                 key="sc_size_mode",
                                 help="Nx×Ny 周期の内接円: 中の格子を Nx×Ny 個並べ、その内接円を円柱にする。"
                                      "直径を指定: 円柱の直径を直接決める。中の格子は円柱を埋めるのに要る数だけ"
                                      "自動で並べる。")

    if size_mode == "inscribed":
        c1, c2, c3 = st.columns([1.4, 1, 1])
        with c1:
            preset = st.radio("周期の数", list(PRESETS), horizontal=True, key="sc_preset")
        if PRESETS[preset] is None:
            with c2:
                nx = int(st.number_input("Nx（a1 方向）", 1, 9, 2, 1, key="sc_nx"))
            with c3:
                ny = int(st.number_input("Ny（a2 方向）", 1, 9, 2, 1, key="sc_ny"))
        else:
            nx, ny = PRESETS[preset]
    else:
        nx = ny = 1   # 直径から下で決める

    spec = {'nx': nx, 'ny': ny, 'cut': bool(cut)}
    if cut:
        gap = 0.0
        if mode == "keep":
            g1, g2 = st.columns([1.3, 1])
            with g1:
                spec['arrangement'] = st.radio(
                    "円柱の並べ方", list(ARRANGEMENT_LABELS), format_func=ARRANGEMENT_LABELS.get, horizontal=True,
                    key="sc_arrangement",
                    help="切り抜いた円柱（周期領域）どうしをどう並べるか。円柱の中はサイドバーの格子のまま。"
                         "円柱の中心どうしの間隔 = 円柱の直径（直径を指定したとき）または並べた周期の内接円の直径"
                         " ＋ 余白。六方最密配置では格子の角度が 60° になる（中の格子が正方でも）。")
            with g2:
                gap = st.number_input("周期領域の外周の余白 [nm]", 0.0, 1e5, 0.0, 10.0, key="sc_gap",
                                      help="0 なら隣の円柱とぴったり並ぶ（円柱どうしが点で接する）。"
                                           "余白を足すと周期が長くなり、孤立した円柱に近づく（余白は外側の材質）。")
        spec.update(mode=mode, gap_nm=float(gap))
        r1, r2, r3, r4 = st.columns(4)
        with r3:
            cx = st.number_input("中心 x [nm]", -1e6, 1e6, 0.0, 1.0, key="sc_cx")
        with r4:
            cy = st.number_input("中心 y [nm]", -1e6, 1e6, 0.0, 1.0, key="sc_cy")
        if size_mode == "diameter":
            with r1:
                diameter = st.number_input("円柱の直径 [nm]", 10.0, 1e5, 1000.0, 10.0, key="sc_diameter",
                                           help="余白 0 なら、隣の円柱と点で接する（中心どうしの間隔 = 直径）。")
            try:
                n = tiles_for_diameter(a1, a2, diameter, float(np.hypot(cx, cy)))
            except ValueError as e:
                st.error(str(e))
                n = 1
            nx = ny = n
            spec.update(nx=n, ny=n, size_mode="diameter", diameter_nm=float(diameter))
            radius, auto = 0.5 * float(diameter), False
            with r2:
                st.metric("中の格子を並べる数", f"{n}×{n}",
                          help="円柱を埋めるのに要る数（自動）。円柱の外に出る分は切り抜きで消える。")
            r_in = cut_radius_limit(a1, a2, spec)
        else:
            r_in = cut_radius_limit(a1, a2, spec)
            with r1:
                auto = st.checkbox("半径 = 内接円 − 1 nm", value=True, key="sc_r_auto",
                                   help="内接円ちょうどだと隣の円柱と点で接するので、1 nm だけ小さくする。")
            with r2:
                radius = (r_in - 1.0 if auto else
                          st.number_input("半径 R [nm]", 1.0, 1e6, float(round(r_in - 1.0)), 5.0, key="sc_r"))
                if auto:
                    st.metric("半径 R", f"{radius:.1f} nm")
        A1, A2 = region_lattice(a1, a2, spec)
        depth = st.number_input("基板に食い込ませる深さ [nm]", 0.0, 1e5, 0.0, 5.0, key="sc_depth",
                                help="0 なら基板の表面より上の層だけを切り抜く。")
        spec.update(outside_material=out, radius_nm=float(radius), r_auto=bool(auto),
                    cx_nm=float(cx), cy_nm=float(cy), subst_depth_nm=float(depth))
        if radius + np.hypot(cx, cy) > r_in + 1e-6:
            st.error(f"円柱（半径 {radius:.1f} nm、中心のずれ {np.hypot(cx, cy):.1f} nm）が上限の円"
                     f"（半径 {r_in:.1f} nm）からはみ出し、隣の周期の円柱と重なるか、並べた形の外に出ます。"
                     "半径を小さくしてください。")
    else:
        A1, A2 = supercell_vectors(a1, a2, nx, ny)
        r_in = inscribed_radius(A1, A2)

    gamma = _unit_pitch(A1, A2)[2]
    factor = abs(A1[0] * A2[1] - A1[1] * A2[0]) / abs(a1[0] * a2[1] - a1[1] * a2[0])   # 周期の面積の比
    what = (f"直径 {spec['diameter_nm']:.0f} nm の円柱・中の格子 {nx}×{ny} 個を並べて切り抜き"
            if spec.get('size_mode') == "diameter" else f"{nx}×{ny} 周期")
    st.caption(f"計算する周期: {np.linalg.norm(A1):.0f} nm × {np.linalg.norm(A2):.0f} nm・角度 {gamma:g}°"
               f"（{what}）・円柱の半径の上限 {r_in:.1f} nm")

    if factor > 1 + 1e-9:
        b1, b2 = st.columns([1, 2])
        with b1:
            base = st.number_input("1×1 で使っていた NumBasis", 9, NUMBASIS_MAX, 49, 2, key="sc_base_nb",
                                   help="周期を広げると回折次数が細かく並ぶので、同じ精度には NumBasis を"
                                        "周期の面積の比（Nx×Ny 周期なら Nx×Ny）だけ増やす。")
        rec, raw = recommended_numbasis(base, factor)
        with b2:
            st.write("")
            st.button(f"NumBasis を {rec} にする（{base} × {factor:.4g}）", key="sc_set_nb",
                      on_click=_set_numbasis, args=(rec,), disabled=int(norder_now) == rec,
                      help=f"S4 の計算時間はおよそ NumBasis の 3 乗（1×1 の約 {factor ** 3:,.0f} 倍）、メモリは 2 乗"
                           f"（約 {factor ** 2:,.0f} 倍）で増える。「⏱ 見積もり・収束確認」タブで確かめるとよい。"
                           "周期が長いと伝搬する回折次数が増えるので、結果は R0 だけでなく R_total も見る。")
        if raw > NUMBASIS_MAX:
            st.warning(f"目安の NumBasis（{raw}）が入力欄の上限 {NUMBASIS_MAX} を超えます。上限で計算して、"
                       "収束確認で精度を確かめてください。")
    return spec

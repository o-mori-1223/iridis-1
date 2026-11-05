"""基板上の2次元周期ナノ構造アレイ（円盤/球）をS4(RCWA)で解くStreamlitアプリ。

`RCWA2D/rcwa_s4_array2d_backend.py`（検証済み、diff機械精度でFresnel極限・
エネルギー保存則に一致）のUIラッパー。R8_koumyou_v4.py（1D列構造）とは別の
新規アプリ——2D周期アレイは1D backendでは原理的に扱えないため
（HANDOVER.md 2026-09-24「方針転換」参照）。

形状は円盤（staircase近似なしに厳密）と球（`sphere_disk_layers()`による
staircase近似、n_slicesで分割数を指定）を切り替えられる。入射角は単一角度と
角度掃引を切り替えられる（`calc_array_rcwa_s4_angle_sweep()`使用）。

角度掃引の結果表示は、参考ツール`Gsolver'S COLOR.html`（プロジェクトルート
直下）と見た目・機能を統一した`RCWA2D/color_display.py`
（koumyouアプリR8_koumyou_v4.py・MEEP3Dのmeep3d_app.pyと共通のコンポーネント）
をそのまま再利用している——スウォッチ(0°/30°/60°の色)＋角度チャート＋
スペクトルチャートの3点セット（2026-09-25追加）。単一角度モードは
koumyouアプリの"Single angle (2D)"と同じく、この統一表示ではなく単純な
波長スペクトル線グラフのまま（スウォッチ等を出す角度掃引固有の機能のため）。

WSL側のS4環境（`~/s4env`）で起動すること:
    source ~/s4env/bin/activate && streamlit run array2d_app.py
"""
import threading
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from RCWA2D.material_spec import resolve_nk_fn, list_nk_materials
from RCWA2D.color_display import render_gsolver_style_display

try:
    from RCWA2D.rcwa_s4_array2d_backend import (
        calc_array_rcwa_s4, calc_array_rcwa_s4_angle_sweep, sphere_disk_layers, HAVE_S4,
    )
except ImportError:
    HAVE_S4 = False

st.set_page_config(page_title="Nanodisk 2D array (S4)", page_icon="🔴", layout="wide")
st.title("基板上ナノ円盤2次元周期アレイ シミュレータ（S4 / RCWA）")
st.caption(
    "配置間隔（ピッチ）・観察角を振って発色特性を評価するための、2次元周期アレイ"
    "専用のRCWAソルバーです。円盤はz方向に断面が変化しないため厳密に（staircase"
    "近似なしに）計算できます。"
)

if not HAVE_S4:
    st.error("S4 をインポートできません。WSL上でconda環境等を有効化してから起動してください。")
    st.stop()

nk_namelist = list_nk_materials()


# ================================================================
# Sidebar — structure & calculation settings
# ================================================================
with st.sidebar:
    st.subheader("格子（配置パターン・配置間隔）")
    lattice_shape = st.radio("格子形状", ["正方格子", "矩形格子"], horizontal=True, key="lattice_shape")
    pitch_x_nm = st.number_input("ピッチ x [nm]", 50.0, 5000.0, 400.0, 10.0, key="pitch_x")
    if lattice_shape == "正方格子":
        pitch_y_nm = pitch_x_nm
        st.caption(f"ピッチ y = {pitch_y_nm:.0f} nm（正方格子なのでxと同一）")
    else:
        pitch_y_nm = st.number_input("ピッチ y [nm]", 50.0, 5000.0, 400.0, 10.0, key="pitch_y")

    st.subheader("形状")
    shape_mode = st.radio("形状", ["円盤 (Disk)", "球 (Sphere)"], horizontal=True, key="shape_mode")
    disk_material = st.selectbox("材質", nk_namelist,
                                   index=nk_namelist.index("TiO2") if "TiO2" in nk_namelist else 0,
                                   key="disk_material")
    disk_radius_nm = st.number_input("半径 [nm]", 5.0, 2000.0, 150.0, 5.0, key="disk_radius")
    if shape_mode == "円盤 (Disk)":
        disk_height_nm = st.number_input("高さ [nm]", 5.0, 2000.0, 100.0, 5.0, key="disk_height")
        n_slices = 1
    else:
        disk_height_nm = 2.0 * disk_radius_nm  # 球は基板に1点で接し、高さ=直径
        st.caption(f"球の高さ（直径） = {disk_height_nm:.0f} nm（半径から自動計算）")
        n_slices = st.number_input(
            "staircase分割数 n_slices", 2, 200, 20, 2, key="n_slices",
            help="球をz方向にn_slices枚の薄い円盤で近似する。多いほど真の球に近づくが"
                 "計算層数もそれだけ増える。目安: 20前後で概ね収束、精密な数値が必要なら"
                 "40-80程度まで増やして収束を確認すること。")
    max_r = min(pitch_x_nm, pitch_y_nm) / 2.0
    if disk_radius_nm >= max_r:
        st.warning(f"半径がピッチの半分（{max_r:.0f}nm）以上です。隣の{'球' if shape_mode.startswith('球') else '円盤'}と重なります。")

    st.subheader("環境（超格子側媒質）")
    env_mode = st.radio("環境の指定方法", ["真空 (n=1)", "大気 (Air.nk)", "カスタム数値"],
                         key="env_mode", horizontal=False)
    if env_mode == "真空 (n=1)":
        nk_env_fn = resolve_nk_fn(1.0)
    elif env_mode == "大気 (Air.nk)":
        nk_env_fn = resolve_nk_fn("Air")
    else:
        ec1, ec2 = st.columns(2)
        with ec1:
            env_n = st.number_input("環境 n", 1.0, 10.0, 1.0, 0.01, key="env_n_custom")
        with ec2:
            env_k = st.number_input("環境 k", 0.0, 10.0, 0.0, 0.01, key="env_k_custom")
        nk_env_fn = resolve_nk_fn(complex(env_n, env_k))

    st.subheader("基板")
    subst_mode = st.radio("基板の指定方法", ["材質一覧から選択", "カスタム数値"],
                           key="subst_mode", horizontal=False)
    if subst_mode == "材質一覧から選択":
        subst_material = st.selectbox("材質", nk_namelist,
                                        index=nk_namelist.index("SiO2") if "SiO2" in nk_namelist else 0,
                                        key="subst_material")
        nk_subst_fn = resolve_nk_fn(subst_material)
    else:
        sc1, sc2 = st.columns(2)
        with sc1:
            subst_n = st.number_input("基板 n", 1.0, 20.0, 1.5, 0.01, key="subst_n_custom")
        with sc2:
            subst_k = st.number_input("基板 k", 0.0, 20.0, 0.0, 0.01, key="subst_k_custom")
        nk_subst_fn = resolve_nk_fn(complex(subst_n, subst_k))

    st.subheader("入射条件")
    angle_mode = st.radio("入射角モード", ["単一角度", "角度掃引"], horizontal=True, key="angle_mode")
    phi_deg = st.number_input("面内回転角 φ [deg]", -180.0, 180.0, 0.0, 1.0, key="phi_deg")
    if angle_mode == "単一角度":
        theta_deg = st.number_input("入射角 θ [deg]", 0.0, 89.0, 0.0, 1.0, key="theta_deg")
    else:
        theta_range = st.slider("入射角 θ の範囲 [deg]", 0.0, 89.0, (0.0, 60.0), 1.0, key="theta_range")
        theta_step = st.number_input("角度ステップ [deg]", 1.0, 45.0, 10.0, 1.0, key="theta_step")
        theta_deg = theta_range[0]  # 単一角度モード用コード(所要時間見積もり等)との互換のため代表値
        n_theta = int(np.floor((theta_range[1] - theta_range[0]) / theta_step + 1e-9)) + 1
        st.caption(f"θ = {theta_range[0]:.0f}° 〜 {theta_range[1]:.0f}°、{theta_step:.0f}°刻みで{n_theta}点")

    st.subheader("波長・計算設定")
    wl_min_nm = st.number_input("波長 min [nm]", 200.0, 1500.0, 380.0, 10.0, key="wl_min")
    wl_max_nm = st.number_input("波長 max [nm]", 200.0, 1500.0, 780.0, 10.0, key="wl_max")
    n_freq = st.number_input("波長点数", 3, 201, 21, 2, key="n_freq")
    norder = st.number_input("NumBasis（次数）", 9, 201, 49, 2, key="norder")


# ================================================================
# Geometry helpers (shared by disks list & structure preview)
# ================================================================
def _build_layers():
    disk_medium_fn = resolve_nk_fn(disk_material)
    if shape_mode == "円盤 (Disk)":
        layers = [{
            'thickness_nm': float(disk_height_nm),
            'bg_material': '__ENV__',
            'disks': [{'material': disk_material, 'cx_nm': 0.0, 'cy_nm': 0.0,
                        'radius_nm': float(disk_radius_nm)}],
        }]
    else:
        layers = sphere_disk_layers(float(disk_radius_nm), disk_material, int(n_slices),
                                      bg_material='__ENV__')
    nk_fn_map = {disk_material: disk_medium_fn}
    return layers, nk_fn_map


def _structure_preview_fig():
    """円盤アレイの構造プレビュー(平面図+断面図)。S4計算は一切行わない
    （形状の確認だけを、計算前に高速に行うためのボタン用）。"""
    fig = go.Figure()
    px_, py_, r_ = pitch_x_nm, pitch_y_nm, disk_radius_nm

    # --- 平面図（左サブプロット相当、x座標を左半分にオフセット） ---
    plan_x0 = 0.0
    fig.add_shape(type="rect", x0=plan_x0, y0=0, x1=plan_x0 + px_, y1=py_,
                   line=dict(color="rgba(0,0,0,0.4)", width=1.5, dash="dot"),
                   fillcolor="rgba(230,230,230,0.4)")
    fig.add_shape(type="circle",
                   x0=plan_x0 + px_/2 - r_, y0=py_/2 - r_,
                   x1=plan_x0 + px_/2 + r_, y1=py_/2 + r_,
                   line=dict(color="#c0392b", width=2), fillcolor="rgba(226,87,44,0.55)")
    fig.add_annotation(x=plan_x0 + px_/2, y=py_ * 1.08, text=f"pitch_x={px_:.0f}nm",
                        showarrow=False, font=dict(size=11))
    fig.add_annotation(x=plan_x0 - px_*0.12, y=py_/2, text=f"pitch_y={py_:.0f}nm",
                        showarrow=False, textangle=-90, font=dict(size=11))
    fig.add_annotation(x=plan_x0 + px_/2, y=py_/2, text=f"r={r_:.0f}nm",
                        showarrow=False, font=dict(size=10, color="white"))
    fig.add_annotation(x=plan_x0 + px_/2, y=-py_*0.12, text="平面図（単位格子1つ分）",
                        showarrow=False, font=dict(size=12))

    # --- 断面図（右側、x座標を右半分にオフセット） ---
    is_sphere = shape_mode.startswith("球")
    cs_x0 = px_ * 1.6
    h_ = disk_height_nm  # 球の場合は2*radius（呼び出し元で計算済み）
    subst_h = max(h_ * 0.6, 40.0)
    env_h = max(h_ * 0.8, 30.0)
    # 基板
    fig.add_shape(type="rect", x0=cs_x0, y0=-subst_h, x1=cs_x0 + px_, y1=0,
                   line=dict(width=1, color="rgba(0,0,0,0.3)"), fillcolor="rgba(160,200,230,0.7)")
    if is_sphere:
        # 球（基板に1点で接し、中心がz=r）は断面図では円として描ける（staircase近似の
        # 元になる真の球形状そのものを見せる——実際の計算はn_slices枚の円盤近似）。
        fig.add_shape(type="circle", x0=cs_x0 + px_/2 - r_, y0=0, x1=cs_x0 + px_/2 + r_, y1=2*r_,
                       line=dict(color="#c0392b", width=2), fillcolor="rgba(226,87,44,0.55)")
        fig.add_shape(type="rect", x0=cs_x0, y0=h_, x1=cs_x0 + px_, y1=h_ + env_h,
                       line=dict(width=0), fillcolor="rgba(240,240,240,0.25)")
    else:
        # 円盤（幅=2r、中心に配置）＋左右は環境
        fig.add_shape(type="rect", x0=cs_x0, y0=0, x1=cs_x0 + px_/2 - r_, y1=h_,
                       line=dict(width=0), fillcolor="rgba(240,240,240,0.5)")
        fig.add_shape(type="rect", x0=cs_x0 + px_/2 + r_, y0=0, x1=cs_x0 + px_, y1=h_,
                       line=dict(width=0), fillcolor="rgba(240,240,240,0.5)")
        fig.add_shape(type="rect", x0=cs_x0 + px_/2 - r_, y0=0, x1=cs_x0 + px_/2 + r_, y1=h_,
                       line=dict(color="#c0392b", width=2), fillcolor="rgba(226,87,44,0.55)")
        fig.add_shape(type="rect", x0=cs_x0, y0=h_, x1=cs_x0 + px_, y1=h_ + env_h,
                       line=dict(width=0), fillcolor="rgba(240,240,240,0.25)")
    fig.add_annotation(x=cs_x0 + px_/2, y=h_/2, text=disk_material,
                        showarrow=False, font=dict(size=11, color="white"))
    fig.add_annotation(x=cs_x0 + px_/2, y=-subst_h/2,
                        text=(subst_material if subst_mode == "材質一覧から選択" else f"n={subst_n:g}"),
                        showarrow=False, font=dict(size=11))
    fig.add_annotation(x=cs_x0 + px_/2, y=h_ + env_h * 0.6,
                        text=(env_mode if env_mode != "カスタム数値" else f"n={env_n:g}"),
                        showarrow=False, font=dict(size=10, color="gray"))
    fig.add_annotation(x=cs_x0 + px_/2, y=-subst_h*1.25, text="断面図（周期境界を点線で表示せず1周期分のみ）",
                        showarrow=False, font=dict(size=12))
    fig.add_shape(type="line", x0=cs_x0, y0=0, x1=cs_x0 + px_, y1=0,
                   line=dict(color="black", width=1))

    fig.update_layout(
        height=380, showlegend=False, plot_bgcolor="white",
        xaxis=dict(visible=False, range=[-px_*0.25, cs_x0 + px_*1.1]),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1,
                    range=[-subst_h*1.5, max(py_, h_+env_h) * 1.25]),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


show_preview = st.button("🔍 形状を表示（計算前プレビュー）", type="secondary")
if show_preview:
    st.session_state["show_preview_fig"] = True
if st.session_state.get("show_preview_fig"):
    st.plotly_chart(_structure_preview_fig(), use_container_width=True)
    if shape_mode.startswith("球"):
        st.caption(
            "上：単位格子の平面図（球1個の最大断面＋周囲の環境）。下：断面図（基板 → 球 → 環境、"
            f"真の球形状を表示）。実際の計算ではこの球をn_slices={int(n_slices)}枚の薄い円盤"
            "（staircase近似）に分割している——分割そのものはこのプレビューには描いていない。"
            "S4計算は行っていない、形状確認専用の高速プレビューです。"
        )
    else:
        st.caption(
            "上：単位格子の平面図（円盤1個＋周囲の環境）。下：断面図（基板 → 円盤層 → 環境）。"
            "S4計算は行っていない、形状確認専用の高速プレビューです。"
        )

st.divider()


# ================================================================
# Time estimate (single-point timing probe, no full sweep)
# ================================================================
def _run_single_point(wl_nm):
    layers, nk_fn_map = _build_layers()
    t0 = time.time()
    calc_array_rcwa_s4(
        np.array([wl_nm]), theta_deg * np.pi / 180.0, phi_deg * np.pi / 180.0,
        pitch_x_nm / 1000.0, pitch_y_nm / 1000.0, int(norder), layers, nk_fn_map,
        nk_env_fn, nk_subst_fn,
    )
    return time.time() - t0


n_theta_total = int(n_theta) if angle_mode == "角度掃引" else 1
n_points_total = int(n_freq) * n_theta_total

ec1, ec2 = st.columns([1, 2])
with ec1:
    if st.button("⏱ 所要時間を見積もる（1点だけ試し計算）"):
        with st.spinner("1点だけ試し計算中..."):
            per_point = _run_single_point((wl_min_nm + wl_max_nm) / 2.0)
        st.session_state["time_estimate"] = per_point
with ec2:
    est = st.session_state.get("time_estimate")
    if est is not None:
        total_est = est * n_points_total
        if angle_mode == "角度掃引":
            detail = f"× {n_freq}波長点 × {n_theta_total}角度点 = {n_points_total}点"
        else:
            detail = f"× {n_freq}波長点"
        st.info(
            f"1点あたり約 {est:.2f} 秒 {detail} ≈ **推定合計 {total_est:.1f} 秒**"
            f"（約 {total_est/60:.1f} 分）。NumBasis={norder}での実測に基づく見積もりです。"
        )

st.divider()


# ================================================================
# Cancellable calculation (background thread + polling reruns)
#
# バックグラウンドスレッドは`st.session_state`に一切触れない設計にしている
# ——`st.session_state.x = y`は内部でScriptRunContextに紐づく処理をするため、
# 別スレッドから呼ぶと（bare/テスト環境で確認済み: AppTestで実行すると
# "missing ScriptRunContext"警告が出て、実際に書き込みが呼び出し側から見えない
# ことがあった）書き込みが失われる場合がある。代わりに、メインスレッドが
# 一度だけ生成した**普通のPython dict**（`calc_state`、session_stateには
# このdictオブジェクトへの参照だけを1回格納する）をスレッドに直接渡し、
# スレッドはそのdictの中身を書き換えるだけにする——dictの要素更新は
# Streamlitを経由しないふつうのPython操作なので、GIL下で安全に別スレッドから
# 行える。読み出す側（メインスレッド、st.session_state.calc_state経由）は
# 同じdictオブジェクトを見ているので、内容は正しく反映される。
#
# 「実行中かどうか」は`threading.Thread.is_alive()`（Pythonが保証する、
# どのスレッドから呼んでも安全なAPI）で判定する。
# 自動再実行(st.rerun())の連鎖も`_AUTO_POLL_LIMIT`回で打ち切り、それ以上は
# 手動更新ボタンに切り替える——バグ等で万一終了が検知できない場合でも、
# ブラウザ側が無限に再実行し続けることを防ぐ安全策。
# ================================================================
_AUTO_POLL_LIMIT = 100  # 0.4秒間隔 x 100 = 約40秒、これを超えたら手動更新に切替

if "calc_state" not in st.session_state:
    st.session_state.calc_state = {
        "thread": None, "stop_event": None,
        "progress": {"done": 0, "total": 0, "elapsed": 0.0},
        "result": None, "swatch_result": None, "wl_ar": None, "theta_ar": None,
        "theta_deg_ar": None, "angle_mode": None, "error": None, "poll_count": 0,
    }
cs = st.session_state.calc_state  # 以降はこのdictオブジェクトを直接読み書きする

_SWATCH_THETA_DEG = np.array([0.0, 30.0, 60.0])
# Gsolver'S COLOR.html互換の角度チャートが表示する固定5波長(RCWA2D/color_display.py
# のtarget_wavelengthsデフォルトと同じ値)。計算する波長グリッドにこれらが厳密に
# 含まれていないと、コンポーネント側が最も近いサンプル波長で代用してしまい
# ラベルがずれる（実測で確認済みの不具合、下の「計算を実行」ハンドラ参照）。
_ANGLE_CHART_TARGET_WL = [400.0, 470.0, 540.0, 600.0, 700.0]


def _worker(wl_ar, theta_ar, is_sweep, layers, nk_fn_map, stop_event, state):
    """バックグラウンドスレッドで実行される。st.session_stateには触れず、
    渡された素のdict(state)だけを更新する（上のコメント参照）。"""
    try:
        progress_cb = lambda done, total, elapsed: state.update(
            progress={"done": done, "total": total, "elapsed": elapsed})
        if is_sweep:
            result = calc_array_rcwa_s4_angle_sweep(
                wl_ar, theta_ar, phi_deg * np.pi / 180.0,
                pitch_x_nm / 1000.0, pitch_y_nm / 1000.0, int(norder), layers, nk_fn_map,
                nk_env_fn, nk_subst_fn, stop_event=stop_event, progress_cb=progress_cb,
            )
        else:
            result = calc_array_rcwa_s4(
                wl_ar, theta_ar[0], phi_deg * np.pi / 180.0,
                pitch_x_nm / 1000.0, pitch_y_nm / 1000.0, int(norder), layers, nk_fn_map,
                nk_env_fn, nk_subst_fn, stop_event=stop_event, progress_cb=progress_cb,
            )
        state["result"] = result

        # 角度掃引の場合のみ、Gsolver'S COLOR.html互換のスウォッチ表示用に
        # 0°/30°/60°を別途計算する（掃引の角度刻みがちょうど30°/60°に当たら
        # ない場合でもスウォッチが欠けないようにするため——koumyouアプリ
        # (R8_koumyou_v4.py)の角度掃引モードと同じ理由・同じ規約）。
        # 停止要求が来ていた場合はこの追加計算を省略する。
        if is_sweep and not (stop_event is not None and stop_event.is_set()):
            state["swatch_result"] = calc_array_rcwa_s4_angle_sweep(
                wl_ar, _SWATCH_THETA_DEG * np.pi / 180.0, phi_deg * np.pi / 180.0,
                pitch_x_nm / 1000.0, pitch_y_nm / 1000.0, int(norder), layers, nk_fn_map,
                nk_env_fn, nk_subst_fn, stop_event=stop_event,
            )
    except Exception as e:  # スレッド内の例外は自動伝播しないので明示的に捕まえて表示する
        state["error"] = str(e)


th_alive = cs["thread"] is not None and cs["thread"].is_alive()

rc1, rc2 = st.columns(2)
with rc1:
    if st.button("▶ 計算を実行", type="primary", disabled=th_alive):
        wl_ar = np.linspace(wl_min_nm, wl_max_nm, int(n_freq))
        is_sweep = (angle_mode == "角度掃引")
        if is_sweep:
            # Gsolver'S COLOR.html互換の角度チャートが表示する固定5波長
            # (400/470/540/600/700nm)が、ユーザーが選んだn_freq/波長範囲の
            # 都合でグリッド上に厳密に乗らない場合（例: 470nmがちょうど
            # サンプル点でない）、コンポーネント側は「最も近いサンプル波長」
            # を代わりに使ってしまい、ラベルが470nmとずれて表示される
            # （実測で確認済み）。これを防ぐため、範囲内に収まる固定波長を
            # 明示的にグリッドへ合流させ、常に厳密一致させる。
            target_wls_in_range = [w for w in _ANGLE_CHART_TARGET_WL if wl_min_nm <= w <= wl_max_nm]
            wl_ar = np.union1d(wl_ar, target_wls_in_range)
            # 度→ラジアン→度の往復は浮動小数点誤差で"29.999999999999996"の
            # ように表示が汚れる（実測で確認済み）。表示用にはこの
            # 変換前の"きれいな"度配列を別に保持し、計算用のラジアン配列とは
            # 混同しない。
            theta_deg_ar_clean = np.linspace(theta_range[0], theta_range[1], n_theta)
            theta_ar = theta_deg_ar_clean * np.pi / 180.0
        else:
            theta_deg_ar_clean = np.array([theta_deg])
            theta_ar = theta_deg_ar_clean * np.pi / 180.0
        layers, nk_fn_map = _build_layers()
        stop_event = threading.Event()
        cs["stop_event"] = stop_event
        cs["wl_ar"] = wl_ar
        cs["theta_ar"] = theta_ar
        cs["theta_deg_ar"] = theta_deg_ar_clean
        cs["angle_mode"] = angle_mode
        cs["result"] = None
        cs["swatch_result"] = None
        cs["error"] = None
        cs["progress"] = {"done": 0, "total": len(wl_ar) * len(theta_ar), "elapsed": 0.0}
        cs["poll_count"] = 0
        th = threading.Thread(target=_worker,
                               args=(wl_ar, theta_ar, is_sweep, layers, nk_fn_map, stop_event, cs),
                               daemon=True)
        cs["thread"] = th
        th.start()
        st.rerun()
with rc2:
    if st.button("⏹ 計算を停止", disabled=not th_alive):
        if cs["stop_event"] is not None:
            cs["stop_event"].set()
        st.warning("停止をリクエストしました。現在計算中の波長点が終わり次第、そこまでの結果で打ち切ります。")

if cs["error"]:
    st.error(f"計算中にエラーが発生しました: {cs['error']}")

if th_alive:
    prog = cs["progress"]
    done, total, elapsed = prog["done"], prog["total"], prog["elapsed"]
    frac = done / total if total else 0.0
    st.progress(frac, text=f"{done}/{total} 波長点  経過 {elapsed:.1f}秒")
    if done > 0:
        eta = elapsed / done * (total - done)
        st.caption(f"推定残り時間: 約 {eta:.1f} 秒")

    if cs["poll_count"] < _AUTO_POLL_LIMIT:
        cs["poll_count"] += 1
        time.sleep(0.4)
        st.rerun()
    else:
        st.info("自動更新を一時停止しました（長時間実行中）。下のボタンで状態を更新できます。")
        if st.button("🔄 状態を更新"):
            cs["poll_count"] = 0
            st.rerun()


# ================================================================
# Results
# ================================================================
result = cs["result"]
if result is not None:
    wl_ar = cs["wl_ar"]
    theta_ar = cs["theta_ar"]
    is_sweep_result = (cs["angle_mode"] == "角度掃引")
    total_pts = int(np.size(result["R0"]))
    n_ok = result["n_computed"]
    if n_ok < total_pts:
        st.warning(f"計算は途中で停止されました（{n_ok}/{total_pts}点まで計算済み。以下は部分的な結果です）。")

    shape_label = (f"球r={disk_radius_nm:.0f}nm(n_slices={int(n_slices)})"
                   if shape_mode.startswith("球") else
                   f"円盤r={disk_radius_nm:.0f}nm/h={disk_height_nm:.0f}nm")
    base_title = f"ピッチ=({pitch_x_nm:.0f},{pitch_y_nm:.0f})nm, {shape_label}, φ={phi_deg:.0f}°"

    if not is_sweep_result:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wl_ar, y=result["R0"], name="R0 (0次/鏡面)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=wl_ar, y=result["R_total"], name="R_total (全伝搬次数)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=wl_ar, y=result["T0"], name="T0 (0次)", mode="lines+markers", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=wl_ar, y=result["T_total"], name="T_total (全伝搬次数)", mode="lines+markers", line=dict(dash="dot")))
        fig.update_layout(
            title=f"θ={cs['theta_deg_ar'][0]:.0f}°, {base_title}",
            xaxis_title="Wavelength (nm)", yaxis_title="R, T",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        theta_deg_ar = cs["theta_deg_ar"]  # 度→ラジアン→度の往復による表示汚れを避け、
                                             # 「計算を実行」時に保存した元の度配列を使う
        quantity = st.selectbox("表示する量", ["R0 (0次/鏡面)", "R_total (全伝搬次数)",
                                                "T0 (0次)", "T_total (全伝搬次数)"],
                                  key="sweep_quantity")
        qkey = {"R0 (0次/鏡面)": "R0", "R_total (全伝搬次数)": "R_total",
                "T0 (0次)": "T0", "T_total (全伝搬次数)": "T_total"}[quantity]
        Z = result[qkey]
        swatch_result = cs["swatch_result"]

        # render_gsolver_style_display は結果をJSONにしてブラウザ側JSへ渡すため、
        # NaN（stop_eventでの途中打ち切り時に未計算分に入る）があると
        # JSON.parse側で構文エラーになる。完全に計算が終わっている場合だけ
        # Gsolver'S COLOR.html互換の統一表示を使い、途中結果はヒートマップに
        # フォールバックする（NaNは欠損として自然に表示できるため）。
        is_complete = (n_ok == total_pts) and (swatch_result is not None) \
            and swatch_result["n_computed"] == np.size(swatch_result["R0"])

        if is_complete:
            render_gsolver_style_display(
                lambdas=wl_ar, thetas=theta_deg_ar, values=Z,
                show_swatches=True,
                swatch_thetas=_SWATCH_THETA_DEG.tolist(),
                swatch_values=swatch_result[qkey],
            )
            st.caption(f"{quantity} — {base_title}")
        else:
            st.info("計算が途中で打ち切られたため、Gsolver'S COLOR.html互換表示ではなく"
                    "簡易ヒートマップで部分結果を表示しています。")
            heat = go.Figure(data=go.Heatmap(
                x=wl_ar, y=theta_deg_ar, z=Z, colorscale="Viridis", zmin=0, zmax=1,
                colorbar=dict(title=quantity),
            ))
            heat.update_layout(
                title=f"{quantity} vs 波長・入射角 — {base_title}",
                xaxis_title="Wavelength (nm)", yaxis_title="入射角 θ (deg)",
                height=420,
            )
            st.plotly_chart(heat, use_container_width=True)

    st.caption(f"basisの次数（実際の計算に使われた項数）: {len(result['order_basis'])}")

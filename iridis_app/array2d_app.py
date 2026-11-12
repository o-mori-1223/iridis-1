"""2Dモード（2次元周期アレイ）の画面。旧 array2d_app.py を関数化したもの。

セッション状態キーは1Dモードと衝突しないよう a2d_angle_mode / a2d_subst_n_custom /
a2d_subst_k_custom 等に改名済み（1D側は元のキー名のまま）。
"""
import copy
import json
import threading
import time
import uuid

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from RCWA2D import s4_parallel
from RCWA2D.color_display import render_gsolver_style_display
from RCWA2D.material_spec import resolve_nk_fn, list_nk_materials
from RCWA2D.cylinder_cut import _unit_pitch as unit_pitch, arrangement_of, cut_structure, diameter_mode, region_lattice
from RCWA2D.rcwa_s4_array2d_backend import (ENV_MATERIAL, SUBST_MATERIAL, calc_array_rcwa_s4,
                                            HAVE_S4 as HAVE_S4_2D)

from .array2d_editor import render_layer_stack_editor, render_step_editor
from .array2d_lens_editor import render_lens_editor
from .array2d_convergence import render_convergence_check
from .array2d_cross_section import stack_cross_section_fig
from .array2d_cut_preview import cut_model_fig, cut_model_obj
from .array2d_model import (build_layers, lattice_vectors_nm, lens_layers, nearest_neighbor_dist_nm,
                            preview_color_map, refined_inputs, stack_label)
from .array2d_preview import incidence_traces, structure_preview_fig
from .array2d_supercell_editor import render_supercell_editor
from .common import (TAB_CHECK, TAB_PREVIEW, TAB_RESULT, TAB_STRUCT, fixed_action_bar_css, main_tabs,
                     show_result_tab, stop_if_no_s4)
from .log import ProgressLog, fmt_bytes, get_logger, heartbeat
from .records import to_jsonable
from .records_ui import consume_autorun, render_restore_notice, render_save_panel

_SWATCH_THETA_DEG = np.array([0.0, 30.0, 60.0])
# Gsolver'S COLOR.html互換の角度チャートが表示する固定5波長(RCWA2D/color_display.py の
# target_wavelengths デフォルトと同じ)。計算グリッドに厳密に含まれていないと、コンポーネント側が
# 最も近いサンプル波長で代用してしまいラベルがずれる（実測で確認済みの不具合）。
_ANGLE_CHART_TARGET_WL = [400.0, 470.0, 540.0, 600.0, 700.0]
_AUTO_POLL_LIMIT = 100  # 0.4秒間隔 x 100 = 約40秒、これを超えたら 2 秒間隔に
_log = get_logger("iridis.2d")


def _wl_grid(p):
    """計算する波長。角度掃引では、角度チャートの固定5波長を波長グリッドへ合流させ、常に厳密一致させる。"""
    wl_ar = np.linspace(p['wl_min_nm'], p['wl_max_nm'], int(p['n_freq']))
    if p['angle_mode'] == "角度掃引":
        wl_ar = np.union1d(wl_ar, [w for w in _ANGLE_CHART_TARGET_WL if p['wl_min_nm'] <= w <= p['wl_max_nm']])
    return wl_ar
_QUANTITIES = {"R0 (0次/鏡面)": "R0", "R_total (全伝搬次数)": "R_total",
               "T0 (0次)": "T0", "T_total (全伝搬次数)": "T_total"}


def _sidebar(nk_namelist):
    """サイドバーの設定を dict で返す（縦に長くならないよう、3 つのタブに分ける）。"""
    p = {}
    with st.sidebar:
        t_lat, t_med, t_light = st.tabs(["格子", "環境・基板", "入射・波長"])
        with t_lat:
            p['lattice_shape'] = st.radio("格子形状", ["正方格子", "矩形格子", "六方格子"],
                                          horizontal=True, key="lattice_shape",
                                          help="六方格子: 単位胞に粒子1個、格子ベクトル間の角度60°"
                                               "（最近接粒子6個が全て距離a＝六方最密配置）。")
            if p['lattice_shape'] == "六方格子":
                px = st.number_input("格子定数 a（最近接粒子間距離）[nm]", 50.0, 5000.0, 400.0, 10.0, key="pitch_x",
                                     help="格子ベクトル a1 = (a, 0)、a2 = (a/2, √3·a/2)。"
                                          "円盤/球が互いに接する最密配置は半径 = a/2 のとき。")
                py, gamma = px, 60.0
            else:
                px = st.number_input("ピッチ x [nm]", 50.0, 5000.0, 400.0, 10.0, key="pitch_x",
                                     help="正方格子ではピッチ y も同じ値。" if p['lattice_shape'] == "正方格子" else None)
                gamma = 90.0
                if p['lattice_shape'] == "正方格子":
                    py = px
                else:
                    py = st.number_input("ピッチ y [nm]", 50.0, 5000.0, 400.0, 10.0, key="pitch_y")
            p.update(pitch_x_nm=px, pitch_y_nm=py, gamma_deg=gamma)

        with t_med:
            env_mode = st.radio("環境（構造の上の媒質）", ["真空 (n=1)", "大気 (Air.nk)", "カスタム数値"],
                                horizontal=True, key="env_mode")
            if env_mode == "真空 (n=1)":
                p['env_spec'] = 1.0
            elif env_mode == "大気 (Air.nk)":
                p['env_spec'] = "Air"
            else:
                ec1, ec2 = st.columns(2)
                with ec1:
                    env_n = st.number_input("環境 n", 1.0, 10.0, 1.0, 0.01, key="env_n_custom")
                with ec2:
                    env_k = st.number_input("環境 k", 0.0, 10.0, 0.0, 0.01, key="env_k_custom")
                p['env_spec'] = complex(env_n, env_k)
            p['nk_env_fn'] = resolve_nk_fn(p['env_spec'])

            st.divider()
            subst_mode = st.radio("基板", ["材質一覧から選択", "カスタム数値"], horizontal=True, key="subst_mode")
            if subst_mode == "材質一覧から選択":
                subst_material = st.selectbox("基板の材質", nk_namelist,
                                              index=nk_namelist.index("SiO2") if "SiO2" in nk_namelist else 0,
                                              key="subst_material")
                p['subst_spec'] = subst_material
                p['nk_subst_fn'] = resolve_nk_fn(subst_material)
                p['subst_label'] = subst_material
            else:
                sc1, sc2 = st.columns(2)
                with sc1:
                    subst_n = st.number_input("基板 n", 1.0, 20.0, 1.5, 0.01, key="a2d_subst_n_custom")
                with sc2:
                    subst_k = st.number_input("基板 k", 0.0, 20.0, 0.0, 0.01, key="a2d_subst_k_custom")
                p['subst_spec'] = complex(subst_n, subst_k)
                p['nk_subst_fn'] = resolve_nk_fn(p['subst_spec'])
                p['subst_label'] = f"n={subst_n:g}"

        with t_light:
            p['angle_mode'] = st.radio("入射角モード", ["単一角度", "角度掃引"], horizontal=True, key="a2d_angle_mode")
            p['phi_deg'] = st.number_input("面内回転角 φ [deg]", -180.0, 180.0, 0.0, 1.0, key="phi_deg",
                                           help="入射面（入射光と基板の法線を含む面）の向き。x 軸から測る。")
            p['pol_deg'] = st.slider("偏光角 ψ [deg]（0 = s 偏光、90 = p 偏光）", -90.0, 90.0, 0.0, 5.0, key="pol_deg",
                                     help="電場の向き。入射面に垂直（s）から入射面内（p）へ ψ だけ回した直線偏光"
                                          "（s 成分 cosψ、p 成分 sinψ）。φ を回すと入射面そのものが回るので、"
                                          "φ と ψ は別物（θ=0° のときだけ、φ を 90° 回すことと ψ=90° が同じになる）。"
                                          "±ψ は s と p の重ね方の符号が違い、異方的な構造では結果が変わることがある。"
                                          "ψ=45° は s と p を同じ強さで重ねた直線偏光で、無偏光（自然光）とは別物"
                                          "（無偏光の反射率は ψ=0° と 90° の結果の平均）。")
            if p['angle_mode'] == "単一角度":
                p['theta_deg'] = st.number_input("入射角 θ [deg]", 0.0, 89.0, 0.0, 1.0, key="theta_deg")
                p['n_theta'] = 1
            else:
                lo, hi = st.slider("入射角 θ の範囲 [deg]", 0.0, 89.0, (0.0, 60.0), 1.0, key="theta_range")
                step = st.number_input("角度ステップ [deg]", 1.0, 45.0, 10.0, 1.0, key="theta_step")
                p['theta_range'] = (lo, hi)
                p['theta_step'] = step
                p['theta_deg'] = lo  # 所要時間見積もり用の代表値
                p['n_theta'] = int(np.floor((hi - lo) / step + 1e-9)) + 1

            st.divider()
            w1, w2 = st.columns(2)
            with w1:
                p['wl_min_nm'] = st.number_input("波長 min [nm]", 200.0, 1500.0, 380.0, 10.0, key="wl_min")
                p['n_freq'] = st.number_input("波長点数", 3, 201, 21, 2, key="n_freq")
            with w2:
                p['wl_max_nm'] = st.number_input("波長 max [nm]", 200.0, 1500.0, 780.0, 10.0, key="wl_max")
                p['norder'] = st.number_input("NumBasis（次数）", 9, s4_parallel.numbasis_max(), 49, 2, key="norder",
                                              help="大きいほど正確だが、計算時間とメモリは急に増える（およそ3乗）。"
                                                   "「⏱ 見積もり・収束確認」のタブで、増やしても結果が変わらないか確かめるとよい。")
            n_wl = len(_wl_grid(p))
            added = n_wl - int(p['n_freq'])
            n_ang = p['n_theta']
            extra_ang = 0
            if p['angle_mode'] == "角度掃引":   # 色見本（スウォッチ）の 0/30/60° も同じ計算で解く
                grid = np.linspace(p['theta_range'][0], p['theta_range'][1], p['n_theta'])
                extra_ang = sum(not np.any(np.isclose(grid, t)) for t in _SWATCH_THETA_DEG)
            notes = ([f"角度チャート用の固定波長 400/470/540/600/700 nm のうち刻みに無い {added} 波長"] if added else []) + \
                    ([f"色見本用の 0/30/60° のうち刻みに無い {extra_ang} 角度"] if extra_ang else [])
            st.caption(f"計算する点: {n_wl} 波長 × {n_ang + extra_ang} 角度 = {n_wl * (n_ang + extra_ang)} 点",
                       help=f"{'、'.join(notes)}を足している。" if notes else None)
    return p


def _calc_args(p):
    """calc_array_rcwa_s4 / s4_parallel.solve_2d に共通の引数（格子・次数・環境/基板）。"""
    return dict(phi_rad=np.radians(p['phi_deg']), pitch_x_um=p['pitch_x_nm'] / 1000.0,
                pitch_y_um=p['pitch_y_nm'] / 1000.0, norder=int(p['norder']),
                nk_env_fn=p['nk_env_fn'], nk_subst_fn=p['nk_subst_fn'], gamma_deg=p['gamma_deg'],
                pol_deg=float(p['pol_deg']))


def _super_vectors(a1, a2, sc):
    """周期領域の格子ベクトル（sc が無ければ単位胞の a1, a2）。余白は「内側を残す」切り抜きのときだけ。
    円柱の並べ方（正方配置／六方最密配置）を選んでいれば、その格子（RCWA2D/cylinder_cut.region_lattice）。"""
    return region_lattice(a1, a2, sc)


def _cell_params(p, sc, a1, a2):
    """計算に使う p。周期領域（sc）を使うときは、周期と角度を周期領域の格子ベクトルに合わせる
    （円柱を六方最密配置にすると、中の格子が正方でも角度は 60°）。"""
    if not sc:
        return p
    A1, A2 = _super_vectors(a1, a2, sc)
    px, py, gamma = unit_pitch(A1, A2)
    return dict(p, pitch_x_nm=px, pitch_y_nm=py, gamma_deg=gamma)


def _build_cut_layers(array_layers, step_regions, step_film, lens, a1, a2, sc):
    """S4 に渡す layers と nk_fn_map。周期領域の設定 sc があれば並べて円柱で切り抜く。"""
    layers, nk_fn_map = build_layers(array_layers, step_regions, step_film, lens=lens, lattice=(a1, a2))
    if sc:
        layers = cut_structure(layers, a1, a2, sc)[0]
        out = sc.get('outside_material')
        if sc.get('cut') and out and out not in (ENV_MATERIAL, SUBST_MATERIAL) and out not in nk_fn_map:
            nk_fn_map[out] = resolve_nk_fn(out)
    return layers, nk_fn_map


def _supercell_label(sc):
    if not sc:
        return ""
    if diameter_mode(sc):
        s = f"直径 {sc['diameter_nm']:.0f} nm の円柱で切り抜き（中の格子 {sc['nx']}×{sc['ny']} 個）"
        return s + {'square': '・円柱は正方配置', 'hex': '・円柱は六方最密配置'}[arrangement_of(sc)]
    s = f"{sc['nx']}×{sc['ny']} 周期"
    if sc.get('cut'):
        s += f"・円柱 R={sc['radius_nm']:.0f} nm で{'切り抜き' if sc['mode'] == 'keep' else '穴あけ'}"
        s += {'square': '・円柱は正方配置', 'hex': '・円柱は六方最密配置'}.get(arrangement_of(sc), '')
    return s


def _worker(wl_ar, theta_ar, is_sweep, layers, nk_fn_map, args, stop_event, state):
    """バックグラウンドスレッド。st.session_state には一切触れず、渡された素の dict
    (state) だけを更新する（別スレッドから st.session_state に書くと、ScriptRunContext が
    無いため呼び出し側から見えないことがあった——AppTestで実測）。
    角度掃引のときは、Gsolver'S COLOR.html互換スウォッチ用の 0/30/60° も同じ並列計算に
    まとめて解き、結果を掃引分とスウォッチ分に分ける。"""
    label = f"2D 計算 {state['calc_id'][:8]}"
    plog = ProgressLog(_log, label)

    def progress(done, total):
        elapsed = time.time() - state["t_start"]
        state.update(progress={"done": done, "total": total, "elapsed": elapsed})
        plog(done, total, elapsed)

    beat = heartbeat(_log, label, lambda: (state["progress"]["done"], state["progress"]["total"],
                                           time.time() - state["t_start"]))
    try:
        sw = np.radians(_SWATCH_THETA_DEG) if is_sweep else np.array([])
        all_theta = list(dict.fromkeys(np.concatenate([theta_ar, sw]).tolist()))
        res = s4_parallel.solve_2d(
            wl_ar, np.array(all_theta), args['phi_rad'], args['pitch_x_um'], args['pitch_y_um'],
            args['norder'], layers, nk_fn_map, args['nk_env_fn'], args['nk_subst_fn'],
            gamma_deg=args['gamma_deg'], stop_event=stop_event, pol_deg=args['pol_deg'], progress_cb=progress)

        def subset(thetas):
            rows = [all_theta.index(t) for t in thetas]
            out = {k: res[k][rows] for k in ('R0', 'R_total', 'T0', 'T_total')}
            out.update(order_basis=res['order_basis'], n_computed=int(res['computed'][rows].sum()))
            return out

        result = subset(list(theta_ar))
        if not is_sweep:
            result = {k: (v[0] if k in ('R0', 'R_total', 'T0', 'T_total') else v) for k, v in result.items()}
        state["result"] = result
        if is_sweep and not stop_event.is_set():
            state["swatch_result"] = subset(sw.tolist())
        n_done, n_all = int(res['computed'].sum()), int(res['computed'].size)
        _log.info("2D 計算 %s: %s（%d/%d 点、%.0f 秒）", state['calc_id'][:8],
                  "停止" if n_done < n_all else "完了", n_done, n_all, time.time() - state["t_start"])
    except Exception as e:  # スレッド内の例外は自動伝播しないので明示的に捕まえて表示する
        _log.exception("2D 計算 %s: エラー", state['calc_id'][:8])
        state["error"] = str(e)
    finally:
        beat.set()


def _record_snapshot(p, step_regions, step_film, lens, nk_fn_map, sc=None):
    """解析記録用に、計算を始めた時点の条件を JSON にできる形で写し取る（あとで画面を変えても記録は変わらない）。
    p はサイドバーの値（1周期の格子）。sc は周期領域と円柱の切り抜きの設定。"""
    try:
        params = to_jsonable({
            "sidebar": {k: v for k, v in p.items() if not k.endswith('_fn')},
            "array_layers": st.session_state.array_layers, "step_regions": step_regions,
            "step_film": step_film, "lens": lens, "supercell": sc,
        })
    except TypeError as e:
        params = {"_error": str(e)}
    lattice = (f"六方 a={p['pitch_x_nm']:.0f}" if p['lattice_shape'] == "六方格子"
               else f"{p['lattice_shape'][:2]} {p['pitch_x_nm']:.0f}×{p['pitch_y_nm']:.0f}")
    theta = (f"θ={p['theta_deg']:g}°" if p['angle_mode'] == "単一角度"
             else f"θ={p['theta_range'][0]:g}〜{p['theta_range'][1]:g}°（{p['n_theta']}点）")
    structure = _structure_label(st.session_state.array_layers, step_regions, lens)
    if sc:
        structure += f"（{_supercell_label(sc)}）"
    text = (f"{lattice} nm, {structure}, 基板 {p['subst_label']}, {theta}, "
            f"φ={p['phi_deg']:g}°, ψ={p['pol_deg']:g}°, λ={p['wl_min_nm']:g}〜{p['wl_max_nm']:g} nm, NumBasis {p['norder']}")
    materials = [m for m in (*nk_fn_map, p['env_spec'], p['subst_spec']) if isinstance(m, str)]
    fig_inputs = copy.deepcopy(dict(array_layers=st.session_state.array_layers, step_regions=step_regions,
                                    step_film=step_film, lens=lens, subst_label=p['subst_label'],
                                    lattice_shape=p['lattice_shape'], supercell=sc,
                                    lattice=lattice_vectors_nm(p['pitch_x_nm'], p['pitch_y_nm'], p['gamma_deg'])))
    return {"params": params, "text": text, "materials": materials, "fig_inputs": fig_inputs,
            "default_title": f"{structure} ({lattice} nm)"}


def _structure_label(array_layers, step_regions, lens):
    """解析記録の題名・条件に使う構造の名前。レンズ状の凹凸や基板段差は層構成（array_layers）に
    入らないので、stack_label だけだと「(層なし)」になってしまう。"""
    extra = []
    if lens is not None:
        spec, recipe = lens
        kind = "多層膜円盤" if spec.get('film_extent') == 'disk' else "形に沿った多層膜"
        extra.append(f"レンズ状{'くぼみ' if spec.get('concave') else '凸'}"
                     f"（a={spec['base_radius_nm']:g}, h={spec['height_nm']:g} nm）＋{kind} {len(recipe)} 段")
    if step_regions:
        extra.append(f"基板段差 {len(step_regions)} 領域")
    if not array_layers and extra:
        return " + ".join(extra)
    return " + ".join([stack_label(array_layers)] + extra)


def _structure_figures(fi):
    """解析記録に添える構造の図（3D と、中心を通る x 方向の縦断面）。fi は計算開始時の写し。"""
    a1, a2 = fi['lattice']
    lens, sc = fi['lens'], fi.get('supercell')
    lens_info, lens_center = None, (0.0, 0.0)
    if lens is not None:
        _, lens_info = lens_layers(lens[0], lens[1], a1, a2)
        lens_center = (lens[0]['cx_nm'], lens[0]['cy_nm'])
    layers_s4, _ = _build_cut_layers(fi['array_layers'], fi['step_regions'], fi['step_film'], lens, a1, a2, sc)
    A1, A2 = _super_vectors(a1, a2, sc)
    color_map = preview_color_map(fi['array_layers'], fi['step_film'])
    extra = {}
    if sc and sc.get('cut'):
        extra["cut_3d"] = cut_model_fig(layers_s4, A1, A2, color_map, fi['subst_label'], _z_bottom(lens_info, sc),
                                        title=f"切り抜いた形の 3D モデル（{_supercell_label(sc)}）")
    return {
        **extra,
        "structure_3d": structure_preview_fig(fi['array_layers'], a1, a2, fi['subst_label'], fi['lattice_shape'],
                                              fi['step_regions'], fi['step_film'], n_side=0,
                                              lens_info=lens_info, lens_center=lens_center, supercell=sc),
        "cross_section": stack_cross_section_fig(
            layers_s4, A1, A2, lens_center[1] if not sc else sc.get('cy_nm', 0.0), color_map, fi['subst_label'],
            n_periods=1, z_bottom_nm=_z_bottom(lens_info, sc),
            lens_info=None if sc else lens_info, lens_center=lens_center, direction='x'),
    }


def _z_bottom(lens_info, sc):
    """断面図の一番下の高さ（凹のレンズでは負、基板に食い込ませる切り抜きでは −深さ）。"""
    z = float(lens_info['slices'][0][0]) if lens_info is not None else 0.0
    if sc and sc.get('cut'):
        z -= float(sc.get('subst_depth_nm', 0.0))   # 切り抜き用に足した基板の層は一番下に積まれる
    return z


def _record_payload(cs):
    """計算が終わった calc_state から、解析記録に保存する中身を作る。"""
    r, sw = cs["result"], cs.get("swatch_result")
    snap = cs["record"]
    result = {"angle_mode": cs["angle_mode"], "wl_nm": cs["wl_ar"], "theta_deg": cs["theta_deg_ar"],
              "pol_deg": cs.get("pol_deg", 0.0), "n_computed": int(r["n_computed"]), "n_total": int(np.size(r["R0"])),
              "n_basis": len(r["order_basis"] or ()), **{k: r[k] for k in ('R0', 'R_total', 'T0', 'T_total')}}
    if sw is not None:
        result["swatch"] = {"theta_deg": _SWATCH_THETA_DEG, **{k: sw[k] for k in ('R0', 'R_total', 'T0', 'T_total')}}
    return {"params": snap["params"], "result": result, "materials": snap["materials"],
            "summary": {"text": snap["text"], "angle_mode": cs["angle_mode"],
                        "partial": result["n_computed"] < result["n_total"]},
            "elapsed_s": cs["progress"]["elapsed"], "calc_id": cs["calc_id"], "calc_time": cs["t_start"],
            "default_title": snap["default_title"],
            "figures_fn": lambda: _structure_figures(snap["fig_inputs"])}


def _calc_controls(p, layers_fn, msg, snapshot_fn=None):
    """計算の実行・停止・進捗表示（バックグラウンドスレッド＋ポーリング再実行）。
    msg は停止・エラーのメッセージを出す場所（結果のタブ）。
    snapshot_fn(nk_fn_map) は計算開始時に呼び、解析記録用の条件の写しを cs["record"] に置く。"""
    if "calc_state" not in st.session_state:
        st.session_state.calc_state = {
            "thread": None, "stop_event": None,
            "progress": {"done": 0, "total": 0, "elapsed": 0.0}, "t_start": 0.0,
            "result": None, "swatch_result": None, "wl_ar": None, "theta_ar": None,
            "theta_deg_ar": None, "angle_mode": None, "error": None, "poll_count": 0,
        }
    cs = st.session_state.calc_state  # 以降はこのdictオブジェクトを直接読み書きする
    th_alive = cs["thread"] is not None and cs["thread"].is_alive()

    # 実行・停止・進み具合は画面右下に固定（どのタブを開いていても見える）。メッセージは結果のタブ（msg）に出す
    fixed_action_bar_css('a2d_calc_action_bar', 300)
    bar = st.container(key='a2d_calc_action_bar')
    with bar:
        rc1, rc2 = st.columns(2)
        with rc1:
            conv_th = st.session_state.get("conv_state", {}).get("thread")
            conv_alive = conv_th is not None and conv_th.is_alive()   # 収束確認と同時に走らせない（メモリ）
            calc_requested = st.button("▶ 計算を実行", type="primary", disabled=th_alive or conv_alive,
                                       width="stretch", on_click=show_result_tab, args=("2d",),
                                       help="収束確認の実行中は使えない。" if conv_alive else None)
            # 「📚 解析記録」の「読み込んで再計算」
            calc_requested = calc_requested or (consume_autorun("2d") and not (th_alive or conv_alive))
        with rc2:
            stop_requested = st.button("⏹ 計算を停止", disabled=not th_alive, width="stretch")

    if calc_requested:
        layers, nk_fn_map = layers_fn()
        if not s4_parallel.fits_in_memory(len(layers) + 2, int(p['norder'])):
            msg.error(f"NumBasis {int(p['norder'])}・S4 の層 {len(layers) + 2} では、1 回の計算に約 "
                      f"{s4_parallel.estimate_task_bytes(len(layers) + 2, int(p['norder'])) / 1e9:.0f} GB 必要で、"
                      "空きメモリに収まりません（解くと WSL ごと落ちる）。NumBasis か層数を減らしてください。")
            calc_requested = False
    if calc_requested:
        is_sweep = (p['angle_mode'] == "角度掃引")
        wl_ar = _wl_grid(p)
        if is_sweep:
            # 表示用の度配列はラジアン変換前の値を保持（度→rad→度の往復で 29.999999 のように汚れるため）
            theta_deg_ar = np.linspace(p['theta_range'][0], p['theta_range'][1], p['n_theta'])
        else:
            theta_deg_ar = np.array([p['theta_deg']])
        theta_ar = np.radians(theta_deg_ar)
        stop_event = threading.Event()
        cs.update(record=snapshot_fn(nk_fn_map) if snapshot_fn else None, calc_id=uuid.uuid4().hex)
        cs.update(stop_event=stop_event, wl_ar=wl_ar, theta_ar=theta_ar, theta_deg_ar=theta_deg_ar,
                  angle_mode=p['angle_mode'], pol_deg=float(p['pol_deg']), result=None, swatch_result=None, error=None, poll_count=0,
                  # 角度掃引では色見本の 0/30/60° も一緒に解く（_worker と同じ数え方。最初から正しい点数を出す）
                  t_start=time.time(), progress={"done": 0, "elapsed": 0.0, "total": len(wl_ar) * len(dict.fromkeys(
                      np.concatenate([theta_ar, np.radians(_SWATCH_THETA_DEG) if is_sweep else []]).tolist()))},
                  concurrency=s4_parallel.max_concurrency(len(layers) + 2, int(p['norder'])))
        rec = cs.get("record") or {}
        _log.info("2D 計算 %s: 開始 %d 点（%d 波長 × %d 角度）・NumBasis %d・S4 の層 %d・%d 並列"
                  "（1 回あたり約 %s）・%s", cs["calc_id"][:8], cs["progress"]["total"], len(wl_ar),
                  cs["progress"]["total"] // len(wl_ar), int(p['norder']), len(layers), cs["concurrency"],
                  fmt_bytes(s4_parallel.estimate_task_bytes(len(layers) + 2, int(p['norder']))), rec.get("text", ""))
        th = threading.Thread(target=_worker, daemon=True,
                              args=(wl_ar, theta_ar, is_sweep, layers, nk_fn_map, _calc_args(p), stop_event, cs))
        cs["thread"] = th
        th.start()
        st.rerun()
    if stop_requested:
        if cs["stop_event"] is not None:
            cs["stop_event"].set()
            _log.info("2D 計算 %s: 停止を受け付けた（実行中の塊が終わり次第打ち切る）", cs["calc_id"][:8])
        msg.warning("停止をリクエストしました。実行中の計算が終わり次第、そこまでの結果で打ち切ります。")

    if cs["error"]:
        msg.error(f"計算中にエラーが発生しました: {cs['error']}")

    if th_alive:
        prog = cs["progress"]
        done, total = prog["done"], prog["total"]
        # 進み具合は計算の塊が終わるたびにしか届かない（NumBasis が大きいと最初の塊まで数分）ので、
        # 経過時間はここで測る。残り時間は塊が終わった時点の値から見積もる
        elapsed = time.time() - cs["t_start"]
        conc = cs.get("concurrency")
        with bar:
            st.progress(done / total if total else 0.0, text=f"{done}/{total} 点  経過 {elapsed:.0f}秒")
            notes = [f"{conc} 並列" + ("（メモリの都合で 1 つずつ）" if conc == 1 else "")] if conc else []
            if done > 0:
                notes.append(f"残り約 {prog['elapsed'] / done * (total - done):.0f} 秒")
            else:
                notes.append("最初の塊を計算中")
            st.caption("・".join(notes))
            # 最初の約 40 秒は 0.4 秒ごと、それ以降は 2 秒ごとに描き直す（長い計算で画面を描き直し続けないように）
            cs["poll_count"] += 1
            time.sleep(0.4 if cs["poll_count"] < _AUTO_POLL_LIMIT else 2.0)
            st.rerun()
    return cs


def _warn_nan_points(Z, theta_deg_ar, wl_ar, swatch_Z):
    """S4 が数値的に解けず NaN になった点を列挙して警告する（表示では補間して埋める）。"""
    pts = [(theta_deg_ar[i], wl_ar[j]) for i, j in zip(*np.nonzero(np.isnan(np.asarray(Z, float))))]
    pts += [(_SWATCH_THETA_DEG[i], wl_ar[j]) for i, j in zip(*np.nonzero(np.isnan(np.asarray(swatch_Z, float))))]
    if not pts:
        return
    listed = ", ".join(f"θ={t:g}°/λ={w:g}nm" for t, w in pts[:12]) + (" …" if len(pts) > 12 else "")
    st.warning(f"S4 が数値的に解けなかった点が {len(pts)} 点あり、表示では同じ角度の隣の波長から"
               f"線形補間しています: {listed}")


def _render_results(cs, p):
    result = cs["result"]
    if result is None:
        return
    wl_ar = cs["wl_ar"]
    total_pts = int(np.size(result["R0"]))
    n_ok = result["n_computed"]
    if n_ok < total_pts:
        st.warning(f"計算は途中で停止されました（{n_ok}/{total_pts}点まで計算済み。以下は部分的な結果です）。")

    lattice_label = (f"六方格子(a={p['pitch_x_nm']:.0f}nm)" if p['lattice_shape'] == "六方格子"
                     else f"ピッチ=({p['pitch_x_nm']:.0f},{p['pitch_y_nm']:.0f})nm")
    pol = cs.get('pol_deg', 0.0)
    pol_label = "s偏光" if pol == 0 else "p偏光" if abs(pol) == 90 else f"偏光角ψ={pol:.0f}°"
    base_title = f"{lattice_label}, {stack_label(st.session_state.array_layers)}, φ={p['phi_deg']:.0f}°, {pol_label}"

    if cs["angle_mode"] != "角度掃引":
        fig = go.Figure()
        for key, name, dash in (("R0", "R0 (0次/鏡面)", None), ("R_total", "R_total (全伝搬次数)", None),
                                ("T0", "T0 (0次)", "dot"), ("T_total", "T_total (全伝搬次数)", "dot")):
            fig.add_trace(go.Scatter(x=wl_ar, y=result[key], name=name, mode="lines+markers",
                                     line=dict(dash=dash) if dash else None))
        fig.update_layout(title=f"θ={cs['theta_deg_ar'][0]:.0f}°, {base_title}",
                          xaxis_title="Wavelength (nm)", yaxis_title="R, T", yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig, width="stretch")
    else:
        theta_deg_ar = cs["theta_deg_ar"]
        quantity = st.selectbox("表示する量", list(_QUANTITIES), key="sweep_quantity")
        qkey = _QUANTITIES[quantity]
        Z = result[qkey]
        swatch_result = cs["swatch_result"]

        # 全点の計算が終わっていれば統一表示（Gsolver'S COLOR.html互換）を使う。S4が数値的に
        # 解けずNaNになった点が残っていても、render_gsolver_style_display が波長方向の補間で
        # 埋めて表示する（どの点を補間したかは警告で示す）。途中停止で未計算の点が多い
        # 場合だけ、欠損をそのまま見せられるヒートマップにする。
        is_complete = (n_ok == total_pts and swatch_result is not None
                       and swatch_result["n_computed"] == np.size(swatch_result["R0"]))

        if is_complete:
            _warn_nan_points(Z, theta_deg_ar, wl_ar, swatch_result[qkey])
            render_gsolver_style_display(lambdas=wl_ar, thetas=theta_deg_ar, values=Z, show_swatches=True,
                                         swatch_thetas=_SWATCH_THETA_DEG.tolist(),
                                         swatch_values=swatch_result[qkey])
            st.caption(f"{quantity} — {base_title}")
        else:
            st.info("計算が途中で打ち切られたため、Gsolver'S COLOR.html互換表示ではなく"
                    "簡易ヒートマップで部分結果を表示しています。")
            heat = go.Figure(data=go.Heatmap(x=wl_ar, y=theta_deg_ar, z=Z, colorscale="Viridis",
                                             zmin=0, zmax=1, colorbar=dict(title=quantity)))
            heat.update_layout(title=f"{quantity} vs 波長・入射角 — {base_title}",
                               xaxis_title="Wavelength (nm)", yaxis_title="入射角 θ (deg)", height=420)
            st.plotly_chart(heat, width="stretch")

    st.caption(f"basisの次数（実際の計算に使われた項数）: {len(result['order_basis'] or ())}")


def _render_preview(p, a1, a2, sc, lens, lens_info, step_regions, step_film, layers_fn, draw):
    """形状プレビュー（S4 計算なし）。左に縦断面、右に 3D。draw=False（タブを開いていない）なら
    表示の設定欄だけ作って図は描かない（設定欄は作らないと値が消える）。"""
    if not (st.session_state.array_layers or step_regions or lens is not None):
        st.info('層を追加すると、ここにプレビューが表示されます。')
        return
    lens_center = (lens[0]['cx_nm'], lens[0]['cy_nm']) if lens else (0.0, 0.0)
    if sc:   # 周期領域を使うときは、断面の既定の位置を円柱の中心に
        lens_center = (sc.get('cx_nm', 0.0), sc.get('cy_nm', 0.0))
    cut = bool(sc and sc.get('cut'))
    v1, v2, v3, v4 = st.columns([2, 2, 1.2, 2.4])
    with v1:
        if sc and cut:
            view = st.radio("表示範囲", ["周期領域 1 つ", "周りの周期領域も"], horizontal=True, key="preview_extent_sc",
                            help="周りの周期領域も: 計算する形（切り抜いた円柱）を格子どおりに隣へ並べて描く"
                                 "（六方最密配置なら中心＋隣の 6 個、正方配置なら 3×3 個）。断面図も 3 周期分になる。"
                                 "3D は「切り抜いた形」の表示のとき。")
        elif sc:
            view = "周期領域"
            st.caption(f"表示範囲: 計算する周期領域（{sc['nx']}×{sc['ny']} 周期）")
        else:
            view = st.radio("表示範囲", ["1周期（単位胞）", "3×3周期"], horizontal=True, key="preview_extent")
    with v2:
        xdir = st.radio("縦断面の向き", ["x 方向に切る", "y 方向に切る"], horizontal=True, key="xsec_dir",
                        help="断面図は S4 に渡す層をそのまま切ったもの（細い横線が層の境目、球・半球・凹凸は階段近似の"
                             "まま）。レンズ状の凹凸では点線が実際の境界面で、点線を階段が十分細かくなぞっていなければ"
                             " Δz を小さくする。横倒しシリンダーの丸い断面は、軸に垂直な方向に切ると見える"
                             "（x 軸のシリンダーなら y 方向）。")
    direction = 'x' if xdir.startswith('x') else 'y'
    with v3:
        pos = st.number_input(f"切る位置 {'y' if direction == 'x' else 'x'} [nm]", -1e6, 1e6,
                              float(lens_center[1] if direction == 'x' else lens_center[0]), 5.0,
                              key=f"xsec_pos_{direction}",
                              help="この位置を通る直線に沿って切る（粒子・凹凸の中心は既定で 0）。")
    with v4:
        kind3d = st.radio("3D の表示", ["切り抜いた形", "切り抜く前＋円柱の壁"], horizontal=True, key="preview_3d_kind",
                          help="切り抜いた形: S4 に渡す層を、層ごとに円・矩形・多角形を厚さ方向に押し出したもの"
                               "（計算する形そのもの。中に別の形を含む形は半透明、空気は描かない）。"
                               "切り抜く前＋円柱の壁: 切り抜く前の形に、切り抜く円柱（赤・半透明）を重ねたもの。") if cut else None
    show_inc = st.toggle(
        f"入射面と入射光を 3D に表示（φ = {p['phi_deg']:g}°）", key="preview_show_inc",
        help="サイドバーの面内回転角 φ の向きの縦の面（黄・半透明）と、入射光（黄）・正反射光（紫）の矢印を重ねる。"
             "φ は x 軸から測り、φ = 0° なら入射光は x の負側の上から入って +x 側へ進む。"
             "角度掃引では範囲の両端の θ を描く（実線が小さい方、破線が大きい方）。")
    if not draw:
        return
    neighbors = view == "周りの周期領域も"
    one_cell = view not in ("3×3周期", "周りの周期領域も")
    try:
        layers_s4, _ = layers_fn()
    except ValueError as e:
        st.error(f"構造を作れません: {e}")
        return
    color_map = preview_color_map(st.session_state.array_layers, step_film)
    A1, A2 = _super_vectors(a1, a2, sc)
    z_bottom = _z_bottom(lens_info, sc)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(stack_cross_section_fig(
            layers_s4, A1, A2, pos, color_map, p['subst_label'], n_periods=1 if one_cell else 3,
            z_bottom_nm=z_bottom, lens_info=None if sc else lens_info, lens_center=lens_center, direction=direction),
            width="stretch")
    with right:
        if kind3d == "切り抜いた形":
            fig3d = cut_model_fig(layers_s4, A1, A2, color_map, p['subst_label'], z_bottom, neighbors=neighbors,
                                  title=f"切り抜いた形の 3D モデル（{_supercell_label(sc)}、S4 に渡す形そのまま"
                                        + ("・周りの周期領域も表示、黒枠 = 計算する周期領域" if neighbors else "") + "）")
        else:
            fig3d = structure_preview_fig(st.session_state.array_layers, a1, a2, p['subst_label'],
                                          p['lattice_shape'], step_regions, step_film,
                                          n_side=0 if one_cell else 1,
                                          lens_info=lens_info, lens_center=lens_center, supercell=sc)
        if show_inc:
            half = 0.5 * max(np.linalg.norm(A1 + A2), np.linalg.norm(A1 - A2)) * \
                (3 if neighbors or not (one_cell or sc) else 1)
            z_top = z_bottom + sum(float(L['thickness_nm']) for L in layers_s4)
            thetas = ([float(p['theta_deg'])] if p['angle_mode'] == "単一角度" else
                      [float(t) for t in dict.fromkeys(p['theta_range'])])
            fig3d.add_traces(incidence_traces(float(p['phi_deg']), thetas, half, z_bottom, z_top,
                                              center=lens_center if sc else (0.0, 0.0)))
        st.plotly_chart(fig3d, width="stretch")
        if cut:
            st.download_button("3D モデルをダウンロード（.obj、単位 nm）",
                               cut_model_obj(layers_s4, A1, A2, p['subst_label'], z_bottom),
                               "iridis_cut_model.obj", "text/plain", key="cut_obj_dl",
                               help="切り抜いた形を Blender などの 3D ソフトで開ける形式で保存する。"
                                    "材質ごとにグループ分けしてある。基板の板は含めない。")


def _render_estimate(p, p_calc, layers_fn):
    """所要時間の見積もり（1点だけ直列で試し計算）。"""
    n_wl = len(_wl_grid(p))
    n_points_total = n_wl * p['n_theta']
    ec1, ec2 = st.columns([1, 2])
    with ec1:
        if st.button("⏱ 所要時間を見積もる（1点だけ試し計算）", width="stretch"):
            with st.spinner("1点だけ試し計算中..."):
                layers, nk_fn_map = layers_fn()
                args = _calc_args(p_calc)
                t0 = time.time()
                calc_array_rcwa_s4(np.array([(p['wl_min_nm'] + p['wl_max_nm']) / 2.0]), np.radians(p['theta_deg']),
                                   args['phi_rad'], args['pitch_x_um'], args['pitch_y_um'], args['norder'],
                                   layers, nk_fn_map, args['nk_env_fn'], args['nk_subst_fn'],
                                   gamma_deg=args['gamma_deg'], pol_deg=args['pol_deg'])
                st.session_state["time_estimate"] = time.time() - t0
                _log.info("所要時間の見積もり: 1 点 %.2f 秒 × %d 点（NumBasis %d・S4 の層 %d）",
                          st.session_state["time_estimate"], n_points_total, args['norder'], len(layers))
    with ec2:
        est = st.session_state.get("time_estimate")
        if est is not None:
            total_est = est * n_points_total
            detail = (f"× {n_wl}波長点 × {p['n_theta']}角度点 = {n_points_total}点"
                      if p['angle_mode'] == "角度掃引" else f"× {n_wl}波長点")
            st.info(f"1点 約 {est:.2f} 秒 {detail} ≈ **直列で {total_est / 60:.1f} 分**（NumBasis {p['norder']}）")
            st.caption("並列で解く本計算はこれより短い", help=(
                f"実際の計算は最大 {s4_parallel.n_workers()} プロセスで並列に解く（実測では概ね 1/2〜1/3。"
                "メモリの都合で並列の数が減ることがある。初回のみワーカー起動に数秒かかる）。"))


def _structure_status(step_regions, lens, sc):
    """「構造」タブの一番上に出す、いま使っている要素の一覧。"""
    n = len(st.session_state.array_layers)
    items = [f"層構成 {n} 層" if n else "層構成なし"]
    if step_regions:
        items.append(f"🪜 基板段差 {len(step_regions)} 領域")
    if lens is not None:
        items.append("🔭 レンズ状の凹凸")
    if sc:
        items.append(f"⭕ {_supercell_label(sc)}")
    return "使用中: " + "・".join(items)


def render_array2d_app():
    st.title("Iridis — 2次元周期アレイ (2D)")
    render_restore_notice("2d")
    stop_if_no_s4(HAVE_S4_2D)

    nk_namelist = list_nk_materials()
    p = _sidebar(nk_namelist)
    a1, a2 = lattice_vectors_nm(p['pitch_x_nm'], p['pitch_y_nm'], p['gamma_deg'])
    nn_dist = nearest_neighbor_dist_nm(a1, a2)
    notice = st.container()   # 構造を作れないときのエラー（どのタブを開いていても見えるようにタブの上）

    t_struct, t_prev, t_check, t_res = main_tabs("2d", [TAB_STRUCT, TAB_PREVIEW, TAB_CHECK, TAB_RESULT])
    with t_struct:
        status = st.container()
        e_layers, e_step, e_lens, e_sc = st.tabs(["層構成", "🪜 基板段差", "🔭 レンズ状の凹凸", "⭕ 周期領域・円柱"])
        with e_layers:
            render_layer_stack_editor(nk_namelist, nn_dist, a1, a2)
        with e_step:
            step_regions = render_step_editor(nk_namelist, a1, a2)
        with e_lens:
            lens = render_lens_editor(nk_namelist, nn_dist, bool(step_regions))
        if lens is not None:
            step_regions = None   # 基板段差とは排他（レンズを優先）
        step_film = lens[1] if lens else (st.session_state.get('step_film_layers', []) if step_regions else [])
        with e_sc:
            sc = render_supercell_editor(nk_namelist, a1, a2, p['norder'])
    p_calc = _cell_params(p, sc, a1, a2)   # 計算に使う周期（周期領域を使うなら Nx·a1, Ny·a2）
    A1, A2 = _super_vectors(a1, a2, sc)

    def layers_fn(array_layers=None, lens_=lens):
        """S4 に渡す layers。周期領域を使うなら並べて切り抜く（同じ入力なら前回の結果を使い回す。
        3×3 などで切り抜きに時間がかかり、計算中は画面が頻繁に描き直されるため）。"""
        array_layers = st.session_state.array_layers if array_layers is None else array_layers
        if not sc:
            return build_layers(array_layers, step_regions, step_film, lens=lens_, lattice=(a1, a2))
        key = json.dumps([array_layers, step_regions, step_film, lens_, sc, a1.tolist(), a2.tolist()],
                         sort_keys=True, default=str)
        cache = st.session_state.get("_cut_layers_cache")
        if cache is None or cache[0] != key:
            cache = (key, _build_cut_layers(array_layers, step_regions, step_film, lens_, a1, a2, sc))
            st.session_state["_cut_layers_cache"] = cache
        return cache[1]

    notes = [_structure_status(step_regions, lens, sc)]
    if sc:
        try:
            cut_layers, _ = layers_fn()
        except ValueError as e:
            notice.error(f"周期領域を作れません: {e}")
            st.stop()
        notes.append(f"S4 に渡す層: {len(cut_layers)} 層（円 {sum(len(L.get('disks', [])) for L in cut_layers)} 個・"
                     f"矩形 {sum(len(L.get('rects', [])) for L in cut_layers)} 個・"
                     f"多角形 {sum(len(L.get('polygons', [])) for L in cut_layers)} 個）")

    lens_info = None
    if lens is not None:
        try:
            lens_layers_list, lens_info = lens_layers(lens[0], lens[1], a1, a2)
            notes.append(f"S4 に渡す凹凸部分: {len(lens_layers_list)} 層（同心円 "
                         f"{sum(len(L['disks']) for L in lens_layers_list)} 個、単位胞の角の島 "
                         f"{sum(len(L['polygons']) for L in lens_layers_list)} 個）")
        except ValueError as e:
            notice.error(f"凹凸の形を作れません: {e}")
            st.stop()
    status.caption("　／　".join(notes))

    with t_prev:
        _render_preview(p, a1, a2, sc, lens, lens_info, step_regions, step_film, layers_fn, draw=t_prev.open)

    with t_check:
        _render_estimate(p, p_calc, layers_fn)
        refined = refined_inputs(st.session_state.array_layers, lens)
        refined_layers_fn = None if refined is None else (lambda: layers_fn(refined[0], refined[1]))
        calc_th = st.session_state.get("calc_state", {}).get("thread")
        render_convergence_check(p_calc, _calc_args(p_calc), layers_fn, refined_layers_fn,
                                 calc_running=calc_th is not None and calc_th.is_alive())

    with t_res:
        msg = st.container()
    cs = _calc_controls(p_calc, layers_fn, msg,
                        snapshot_fn=lambda nk_fn_map: _record_snapshot(p, step_regions, step_film, lens, nk_fn_map, sc))
    th = cs["thread"]
    running = th is not None and th.is_alive()
    with t_res:
        if cs["result"] is None and not running and not cs["error"]:
            st.info("右下の「▶ 計算を実行」を押すと、ここに結果が出ます。")
        _render_results(cs, p)
        if sc and cs["result"] is not None:
            st.caption(f"{_supercell_label(sc)}（周期 {np.linalg.norm(A1):.0f}×{np.linalg.norm(A2):.0f} nm）・"
                       "R_total も確認", help="周期が長いと伝搬する回折次数が増えるので、R0（鏡面反射）だけでなく "
                                              "R_total（全次数の合計）も見る。")
        if cs["result"] is not None and cs.get("record") and not running:
            render_save_panel("2d", _record_payload(cs))

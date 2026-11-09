"""2Dモード「レンズ状の凹凸（形に沿った多層膜・多層膜円盤）」の編集UI。
膜の範囲（全面／円盤）・形状・寸法・膜の付き方・スライス厚さと、成膜レシピ（基板段差と同じエディタ）を編集し、
RCWA2D.curved_multilayer に渡す仕様（dict）を返す。"""
import streamlit as st

from .array2d_editor import render_film_recipe

SHAPE_LABELS = {'spherical_cap': '球冠', 'paraboloid': '放物面', 'cosine': 'コサイン形'}
DEPOSITION_LABELS = {'normal': '形に垂直な方向に一定（スパッタ・ALD 型）',
                     'vertical': '真上から積もる（蒸着型）'}
EXTENT_LABELS = {'full': '全面（多層膜）', 'disk': '円盤（多層膜円盤）'}


def _default_spec():
    return {'shape': 'spherical_cap', 'concave': False, 'base_radius_nm': 150.0, 'height_nm': 100.0,
            'cx_nm': 0.0, 'cy_nm': 0.0, 'deposition': 'normal', 'dz_max_nm': 10.0,
            'film_extent': 'full', 'disk_radius_nm': 150.0}


def _default_recipe():
    return [{'id': 1, 'type': 'simple', 'material': 'TiO2', 'thickness': 51.0},
            {'id': 2, 'type': 'pair', 'mat_a': 'TiO2', 'th_a': 51.0, 'mat_b': 'SiO2', 'th_b': 81.0, 'n_pairs': 7}]


def render_lens_editor(nk_namelist, nn_dist_nm, step_enabled):
    """有効なら（spec, film_recipe）を返す。無効なら None。"""
    enabled = st.checkbox(
        "基板にレンズ状の凸（またはくぼみ）を作り、その上に成膜レシピを形に沿って積む",
        value=False, key="lens_enabled",
        help="軸対称（真上から見て円形）の凹凸を各単位胞の中に1つ置き、膜を形に沿って積む。"
             "S4 には水平に輪切りにした「同心円の層」の積み重ねとして渡す（階段近似）。")
    if not enabled:
        return None
    if step_enabled:
        st.error("「基板段差」と「レンズ状の凹凸」は同時に使えません。どちらか一方をオフにしてください"
                 "（いまはレンズ状の凹凸を使って計算します）。")

    if 'lens_spec' not in st.session_state:
        st.session_state.lens_spec = _default_spec()
        st.session_state.lens_film_layers = _default_recipe()
        st.session_state._lens_layer_id = 2
    spec = st.session_state.lens_spec
    for k, v in _default_spec().items():   # 古いセッションの仕様に新しい項目を補う
        spec.setdefault(k, v)

    e1, e2 = st.columns([2, 1])
    with e1:
        spec['film_extent'] = st.radio(
            "膜の範囲", list(EXTENT_LABELS), format_func=EXTENT_LABELS.get, horizontal=True,
            index=list(EXTENT_LABELS).index(spec['film_extent']), key="lens_extent",
            help="全面: 膜が単位胞全体を覆う。円盤: 形に沿って積んだ膜を半径 R の円筒で垂直に切り取った"
                 "多層膜円盤（外側は基板の表面が環境に出る）。")
    if spec['film_extent'] == 'disk':
        with e2:
            spec['disk_radius_nm'] = st.number_input("円盤の半径 R [nm]", 10.0, 5000.0, spec['disk_radius_nm'],
                                                     5.0, key="lens_disk_r",
                                                     help="底面の半径と同じなら、膜が凹凸の部分だけに載る。")

    c1, c2, c3 = st.columns(3)
    with c1:
        spec['shape'] = st.selectbox("形状", list(SHAPE_LABELS), format_func=SHAPE_LABELS.get,
                                     index=list(SHAPE_LABELS).index(spec['shape']), key="lens_shape",
                                     help="球冠: 球の一部（高さ ≤ 底面の半径、高さ=半径で半球）。"
                                          "放物面: h = H(1-(r/a)²)。コサイン形: 裾がなめらかに 0 になる形。")
    with c2:
        concave = st.radio("向き", ["凸（レンズ）", "凹（くぼみ）"], horizontal=True,
                           index=1 if spec['concave'] else 0, key="lens_concave")
        spec['concave'] = concave.startswith("凹")
    with c3:
        spec['deposition'] = st.selectbox("膜の付き方", list(DEPOSITION_LABELS), format_func=DEPOSITION_LABELS.get,
                                          index=list(DEPOSITION_LABELS).index(spec['deposition']), key="lens_dep")

    d1, d2, d3 = st.columns(3)
    with d1:
        spec['base_radius_nm'] = st.number_input("底面の半径 a [nm]", 5.0, 5000.0, spec['base_radius_nm'], 5.0,
                                                 key="lens_a")
    with d2:
        spec['height_nm'] = st.number_input("深さ [nm]" if spec['concave'] else "高さ [nm]", 1.0, 5000.0,
                                            spec['height_nm'], 5.0, key="lens_h")
    with d3:
        spec['dz_max_nm'] = st.number_input("スライス厚さの上限 Δz [nm]", 1.0, 50.0, spec['dz_max_nm'], 1.0,
                                            key="lens_dz",
                                            help="小さいほど曲面を細かく近似するが、層が増えて遅くなる。"
                                                 "10nm から始め、半分にして結果が変わらないか確かめるとよい。")
    p1, p2 = st.columns(2)
    with p1:
        spec['cx_nm'] = st.number_input("位置 x [nm]", -1e6, 1e6, spec['cx_nm'], 1.0, key="lens_cx")
    with p2:
        spec['cy_nm'] = st.number_input("位置 y [nm]", -1e6, 1e6, spec['cy_nm'], 1.0, key="lens_cy")

    if spec['shape'] == 'spherical_cap' and spec['height_nm'] > spec['base_radius_nm']:
        st.error("球冠の高さ（深さ）は底面の半径以下にしてください（半球まで）。")

    render_film_recipe(nk_namelist, 'lens_film_layers', '_lens_layer_id', 'lens',
                       "**成膜レシピ**（下から上）",
                       "成膜レシピが空なので、凹凸（基板と同じ材質）だけを計算します。")

    if spec['base_radius_nm'] > nn_dist_nm / 2.0 + 1e-6:
        st.error(f"底面の半径（{spec['base_radius_nm']:.0f} nm）が最近接の格子点間距離の半分"
                 f"（{nn_dist_nm / 2.0:.0f} nm）を超え、隣の凹凸と重なります。半径を小さくするか周期を広げてください。")
    if spec['film_extent'] == 'disk' and spec['disk_radius_nm'] > nn_dist_nm / 2.0 + 1e-6:
        st.error(f"円盤の半径（{spec['disk_radius_nm']:.0f} nm）が最近接の格子点間距離の半分"
                 f"（{nn_dist_nm / 2.0:.0f} nm）を超え、隣の円盤と重なります。")
    return dict(spec), list(st.session_state.lens_film_layers)

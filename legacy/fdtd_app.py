"""Structural color simulator -- FDTD backend (Streamlit UI).

Standalone sibling to R8_koumyou_v3.py: reuses the exact same "column builder" input
model (columns_to_layers from RCWA2D/structure_builder.py) so a structure can be typed
in once and compared between the RCWA (R8_koumyou_v3.py) and FDTD (this app) backends.

FDTD is far more expensive per run than RCWA (a broadband time-domain simulation vs. a
single per-wavelength eigenmode solve), so unlike the RCWA app this one does NOT
recompute on every widget tweak: press "FDTD計算を実行" when the structure is ready.
Also unlike the RCWA app, only normal incidence and a single polarization (TE / s-pol,
which equals p-pol at normal incidence for an isotropic structure) are supported --
see fdtd2d/engine.py.
"""
import datetime
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import colour

from RCWA2D.structure_builder import (
    ENV_MATERIAL, SUBST_MATERIAL, columns_to_layers, _expand_to_bands, _col_total_height,
)
from fdtd2d.simulate import run_spectrum
from fdtd2d.color import spectrum_to_srgb

st.set_page_config(
    page_title="Structural color simulator (FDTD)",
    page_icon="🌈",
    layout="wide",
)


@st.cache_data
def get_nk_list():
    nk_list = []
    nk_dirs = "data//nk"
    for f in os.listdir(nk_dirs):
        if os.path.isfile(os.path.join(nk_dirs, f)):
            nk_list.append(os.path.splitext(os.path.basename(f))[0])
    nk_list.sort()
    return nk_list


# ---------------------------------------------------------------------------
# Structure visualization (copied from R8_koumyou_v3.py -- pure functions, no UI state)
# ---------------------------------------------------------------------------
def create_structure_figure(layers, pitch_nm, nk_subst_name, col_widths=None):
    shapes, annotations = [], []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    uniq_mat = sorted({m for L in layers for m, w in L['segments']
                       if m not in (ENV_MATERIAL, SUBST_MATERIAL)})
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(uniq_mat)}
    env_color, subst_color = 'rgba(240,240,240,0.95)', 'rgba(160,200,230,0.9)'

    total_th = sum(L['thickness'] for L in layers)
    subst_th = total_th * 0.2 if total_th > 0 else 50

    shapes.append(go.layout.Shape(type="rect", xref="x", yref="y",
                                   x0=0, y0=-subst_th, x1=pitch_nm, y1=0,
                                   fillcolor="lightgrey", line_width=0))
    annotations.append(dict(x=pitch_nm / 2, y=-subst_th / 2, text=nk_subst_name,
                             showarrow=False, font=dict(color="black")))

    current_y = 0.0
    for L in reversed(layers):
        d, cursor = L['thickness'], 0.0
        for mat, w in L['segments']:
            x0, x1 = pitch_nm * cursor, pitch_nm * (cursor + w)
            clr = env_color if mat == ENV_MATERIAL else (subst_color if mat == SUBST_MATERIAL
                                                           else color_map.get(mat, '#cccccc'))
            shapes.append(go.layout.Shape(type="rect", xref="x", yref="y",
                                           x0=x0, y0=current_y, x1=x1, y1=current_y + d,
                                           fillcolor=clr, line=dict(color="black", width=1)))
            if mat not in (ENV_MATERIAL, SUBST_MATERIAL) and (x1 - x0) > pitch_nm * 0.04:
                annotations.append(dict(x=(x0 + x1) / 2, y=current_y + d / 2,
                                         text=f"{mat}<br>{d:g}nm", showarrow=False,
                                         font=dict(color="white", size=10)))
            cursor += w
        current_y += d

    if col_widths:
        tw, cx = sum(col_widths), 0.0
        for cw in col_widths[:-1]:
            cx += cw
            xp = pitch_nm * cx / tw
            shapes.append(go.layout.Shape(type="line", xref="x", yref="y",
                                           x0=xp, y0=-subst_th, x1=xp, y1=current_y,
                                           line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot")))

    fig = go.Figure()
    fig.update_layout(
        title="Film Stack Visualization",
        xaxis_title="Position (nm)", yaxis_title="Height (nm)",
        xaxis=dict(range=[0, pitch_nm], showgrid=False, zeroline=False),
        yaxis=dict(range=[-subst_th, current_y * 1.1 if current_y > 0 else 100],
                   scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False),
        shapes=shapes, annotations=annotations, plot_bgcolor='white')
    return fig


def get_structure_breakdown_df(columns):
    col_bands = [_expand_to_bands(col, flat_align_height=0.0) for col in columns]
    heights = set([0.0])
    for bands in col_bands:
        for h_lo, h_hi, *_ in bands:
            heights.add(h_lo); heights.add(h_hi)
    heights = sorted(heights)
    data = []
    for i in range(len(heights) - 1):
        h_lo, h_hi = heights[i], heights[i + 1]
        h_mid = (h_lo + h_hi) / 2
        row = {'高さ [nm]': f'{h_lo:.1f}〜{h_hi:.1f}', '厚さ [nm]': round(h_hi - h_lo, 1)}
        for ci, (col, bands) in enumerate(zip(columns, col_bands)):
            mat = ENV_MATERIAL
            for blo, bhi, bmat, *_ in bands:
                if blo <= h_mid < bhi:
                    mat = bmat; break
            row[f'C{ci+1}({col["width"]:.0f}nm)'] = ('空気' if mat == ENV_MATERIAL else
                                                       ('基板' if mat == SUBST_MATERIAL else mat))
        data.append(row)
    return pd.DataFrame(data)


def _columns_key(columns):
    """Hashable representation of `columns` for st.cache_data."""
    key = []
    for c in columns:
        fl_key = tuple(tuple(sorted(fl.items())) for fl in c['film_layers'])
        key.append((c['width'], c['subst_raise'], fl_key))
    return tuple(key)


def _columns_from_key(key):
    return [{'width': w, 'subst_raise': sr, 'film_layers': [dict(items) for items in fl_key]}
            for w, sr, fl_key in key]


@st.cache_data(show_spinner=False)
def _run_fdtd_cached(columns_key, nk_subst_name, n_env, wl_min, wl_max, n_wl, dx_nm, n_poles):
    columns = _columns_from_key(columns_key)
    return run_spectrum(columns, nk_subst_name, n_env, wl_min, wl_max,
                         n_wl=n_wl, dx_nm=dx_nm, n_poles=n_poles)


# ================================================================
# UI
# ================================================================
st.title('Structural color simulator (FDTD)')
st.caption(
    'RCWA版 (R8_koumyou_v3.py) と同じ列ベース構造ビルダーを使います。'
    '正入射・単一偏光 (TE/s-pol、正入射では p-pol と等価) のみ対応。'
    '計算が重いので、構造を決めたら「FDTD計算を実行」を押してください。'
)

nk_namelist = get_nk_list()
nk_idx_subst = nk_namelist.index('Silicon') if 'Silicon' in nk_namelist else 0
nk_idx_film = nk_namelist.index('SiO2') if 'SiO2' in nk_namelist else 0

st.sidebar.header('Light parameters')
spMenu = ('Visible[380-780nm]', 'UV[200-400nm]', 'NIR[700-1000nm]', 'All[200-1000nm]', 'Any')
wl_option = st.sidebar.selectbox('Spectrum range', spMenu)
if wl_option == spMenu[0]:
    wl_min, wl_max = 380.0, 780.0
elif wl_option == spMenu[1]:
    wl_min, wl_max = 200.0, 400.0
elif wl_option == spMenu[2]:
    wl_min, wl_max = 700.0, 1000.0
elif wl_option == spMenu[3]:
    wl_min, wl_max = 200.0, 1000.0
else:
    wl_min, wl_max = st.sidebar.slider('Wavelength range [nm]', 200.0, 1000.0, (400.0, 800.0), 20.0)

st.sidebar.header('Atmosphere / Substrate')
n_env = st.sidebar.number_input('Refractive index (air:1.00)', 1.0, 3.0, 1.0, 0.01, format='%3.2f')
nk_subst_name = st.sidebar.selectbox('Substrate', nk_namelist, index=nk_idx_subst)

st.sidebar.header('FDTD 計算設定')
precision = st.sidebar.radio('精度モード', ['粗め・高速 (推奨)', '高精度・低速'], key='fdtd_precision')
if precision.startswith('粗め'):
    default_dx, default_nwl, default_npoles = 12.0, 21, 3
else:
    default_dx, default_nwl, default_npoles = 5.0, 41, 5
with st.sidebar.expander('詳細設定 (上級者向け)'):
    dx_nm = st.number_input('グリッド幅 dx [nm] (小さいほど高精度・低速)', 1.0, 30.0, default_dx, 0.5)
    n_wl = st.number_input('波長サンプル数', 5, 101, default_nwl, 2)
    n_poles = st.number_input('分散フィットの極数', 1, 8, default_npoles, 1,
                               help='材料のn,kデータに合わせる分散モデルの複雑さ。金属は3以上推奨。')
st.sidebar.caption(
    f'目安の計算時間: {"数十秒〜1分程度" if precision.startswith("粗め") else "数分程度"}'
    '（構造の大きさ・材料数により変動）'
)

# ================================================================
# Column builder (same data model as R8_koumyou_v3.py)
# ================================================================
st.subheader('Structure — Column builder')
st.caption(
    '周期を「列」に分割し、各列で基板の上げ高さとフィルム層を独立に設定します。'
    '列の幅の合計が自動的にピッチになります（等形成膜モードのみ対応）。'
)

if 'fdtd_columns' not in st.session_state:
    st.session_state.fdtd_columns = []
    st.session_state._fdtd_ncol_id = 0
    st.session_state._fdtd_nlayer_id = 0


def _new_col_id():
    st.session_state._fdtd_ncol_id += 1
    return st.session_state._fdtd_ncol_id


def _new_layer_id():
    st.session_state._fdtd_nlayer_id += 1
    return st.session_state._fdtd_nlayer_id


if st.button('＋ 列を追加 (Add column)', key='fdtd_add_col_btn'):
    st.session_state.fdtd_columns.append({
        'id': _new_col_id(), 'width': 300.0, 'subst_raise': 0.0, 'film_layers': [],
    })

_col_rm = _col_mv = _lrm = _ladd = None

for ci, col in enumerate(st.session_state.fdtd_columns):
    cid = col['id']
    total_h = _col_total_height(col)
    label = f"Column {ci+1}  │  幅 {col['width']:.0f} nm  │  高さ {total_h:.0f} nm"
    with st.expander(label, expanded=True):
        cc1, cc2 = st.columns(2)
        with cc1:
            col['width'] = st.number_input('幅 Width [nm]', 1.0, 1e6, col['width'], 1.0,
                                            format='%g', key=f'fcw_{cid}')
        with cc2:
            col['subst_raise'] = st.number_input('基板上げ高さ [nm]', 0.0, 1e6, col['subst_raise'],
                                                  1.0, format='%g', key=f'fcsr_{cid}')
        if col['subst_raise'] > 0:
            st.caption(f'▸ {nk_subst_name}（基板材質）: 0 → {col["subst_raise"]:.0f} nm')

        st.markdown('**フィルム層 Film layers — 下から上 (bottom → top)**')
        for li, fl in enumerate(col['film_layers']):
            lid = fl['id']
            ftype = fl.get('type', 'simple')
            if ftype == 'simple':
                lc1, lc2, lc3 = st.columns([3, 2, 1])
                with lc1:
                    idx_ = nk_namelist.index(fl['material']) if fl['material'] in nk_namelist else nk_idx_film
                    fl['material'] = st.selectbox(f'材質 #{li+1}', nk_namelist, index=idx_, key=f'flm_{cid}_{lid}')
                with lc2:
                    fl['thickness'] = st.number_input(f'厚さ #{li+1} [nm]', 0.0, 1e6,
                                                       fl.get('thickness', 100.0), 1.0,
                                                       format='%g', key=f'flt_{cid}_{lid}')
                with lc3:
                    st.write('')
                    if st.button('✕', key=f'flrm_{cid}_{lid}'):
                        _lrm = (ci, li)
            elif ftype == 'pair':
                st.markdown(f'*🔄 ペア繰り返し層 #{li+1}*')
                st.caption('A = 上側（空気側）/ B = 下側（基板側）')
                p1, p2, p3 = st.columns([2, 2, 1])
                with p1:
                    ia_ = nk_namelist.index(fl.get('mat_a', nk_namelist[nk_idx_film])) if fl.get('mat_a', '') in nk_namelist else nk_idx_film
                    fl['mat_a'] = st.selectbox(f'材質A 上側 #{li+1}', nk_namelist, index=ia_, key=f'fpma_{cid}_{lid}')
                    fl['th_a'] = st.number_input(f'厚さA #{li+1} [nm]', 0.0, 1e6, fl.get('th_a', 50.0),
                                                  1.0, format='%g', key=f'fpta_{cid}_{lid}')
                with p2:
                    ib_ = nk_namelist.index(fl.get('mat_b', nk_namelist[0])) if fl.get('mat_b', '') in nk_namelist else 0
                    fl['mat_b'] = st.selectbox(f'材質B 下側 #{li+1}', nk_namelist, index=ib_, key=f'fpmb_{cid}_{lid}')
                    fl['th_b'] = st.number_input(f'厚さB #{li+1} [nm]', 0.0, 1e6, fl.get('th_b', 50.0),
                                                  1.0, format='%g', key=f'fptb_{cid}_{lid}')
                with p3:
                    fl['n_pairs'] = st.number_input(f'繰り返し #{li+1}', 1, 200, fl.get('n_pairs', 3),
                                                     1, format='%d', key=f'fpn_{cid}_{lid}')
                    st.caption(f'合計 {fl["n_pairs"] * (fl["th_a"] + fl["th_b"]):.0f} nm')
                    if st.button('✕', key=f'fplrm_{cid}_{lid}'):
                        _lrm = (ci, li)
            st.divider()

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button('＋ 単層', key=f'fladd_{cid}'):
                _ladd = (ci, 'simple')
        with bc2:
            if st.button('＋ ペア繰り返し', key=f'fladd_p_{cid}'):
                _ladd = (ci, 'pair')

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            if st.button('◀ 左へ', key=f'fcl_{cid}', disabled=(ci == 0)):
                _col_mv = (ci, -1)
        with mc2:
            if st.button('▶ 右へ', key=f'fcr_{cid}', disabled=(ci == len(st.session_state.fdtd_columns) - 1)):
                _col_mv = (ci, 1)
        with mc3:
            if st.button('✕ この列を削除', key=f'fcrm_{cid}'):
                _col_rm = ci

if _lrm is not None:
    ci, li = _lrm
    st.session_state.fdtd_columns[ci]['film_layers'].pop(li)
    st.rerun()
if _ladd is not None:
    ci, ltype = _ladd
    if ltype == 'simple':
        st.session_state.fdtd_columns[ci]['film_layers'].append({
            'id': _new_layer_id(), 'type': 'simple', 'material': nk_namelist[nk_idx_film], 'thickness': 100.0})
    else:
        st.session_state.fdtd_columns[ci]['film_layers'].append({
            'id': _new_layer_id(), 'type': 'pair',
            'mat_a': nk_namelist[nk_idx_film], 'th_a': 50.0,
            'mat_b': nk_namelist[nk_idx_subst], 'th_b': 50.0, 'n_pairs': 3})
    st.rerun()
if _col_rm is not None:
    st.session_state.fdtd_columns.pop(_col_rm)
    st.rerun()
if _col_mv is not None:
    ci, d = _col_mv; j = ci + d
    st.session_state.fdtd_columns[ci], st.session_state.fdtd_columns[j] = \
        st.session_state.fdtd_columns[j], st.session_state.fdtd_columns[ci]
    st.rerun()

col_widths = [c['width'] for c in st.session_state.fdtd_columns]
pitch_nm = sum(col_widths) if col_widths else 500.0
layers = columns_to_layers(st.session_state.fdtd_columns, flat_align=False)

if st.session_state.fdtd_columns:
    st.caption(f'ピッチ = {pitch_nm:.1f} nm ／ 層数 = {len(layers)}')
else:
    st.info('「列を追加」ボタンで列を追加してください。')

st.subheader('Film Stack Visualization')
st.plotly_chart(create_structure_figure(layers, pitch_nm, nk_subst_name, col_widths=col_widths),
                 use_container_width=True)

with st.expander('層テーブル', expanded=False):
    if st.session_state.fdtd_columns:
        st.dataframe(get_structure_breakdown_df(st.session_state.fdtd_columns),
                     use_container_width=True, hide_index=True)

# ================================================================
# FDTD run
# ================================================================
st.subheader('FDTD Spectrum')

run_clicked = st.button('▶ FDTD計算を実行', type='primary', key='fdtd_run_btn',
                         disabled=(len(st.session_state.fdtd_columns) == 0))

if run_clicked:
    with st.spinner('FDTDを計算中... (構造・解像度によっては数分かかります)'):
        result = _run_fdtd_cached(_columns_key(st.session_state.fdtd_columns), nk_subst_name,
                                   n_env, wl_min, wl_max, int(n_wl), float(dx_nm), int(n_poles))
    st.session_state['fdtd_result'] = result

result = st.session_state.get('fdtd_result')
if result is None:
    st.info('構造を設定して「FDTD計算を実行」を押してください。')
else:
    st.caption(f'グリッド: {result.grid.nx} x {result.grid.nz} セル ／ dx = {result.grid.dx*1e9:.1f} nm')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.wl_nm, y=result.R, name='R', mode='lines'))
    fig.add_trace(go.Scatter(x=result.wl_nm, y=result.T, name='T', mode='lines'))
    fig.add_trace(go.Scatter(x=result.wl_nm, y=result.A, name='A', mode='lines'))
    fig.update_layout(title='FDTD reflectance / transmittance / absorptance',
                       xaxis_title='Wavelength (nm)', yaxis_title='R / T / A')
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    if wl_option == spMenu[0]:
        st.subheader('Colorimetry')
        hex_color, RGB, XYZ = spectrum_to_srgb(result.wl_nm, result.R)
        col1, col2 = st.columns(2)
        with col1:
            st.color_picker('反射色 R', hex_color, key='fdtd_color_' + hex_color)
        with col2:
            st.write('XYZ', XYZ)
            st.write('RGB', RGB)

    df_spec = pd.DataFrame({'Wavelength(nm)': result.wl_nm, 'R': result.R,
                             'T': result.T, 'A': result.A}).set_index('Wavelength(nm)')
    t_delta = datetime.timedelta(hours=9)
    d = datetime.datetime.now(datetime.timezone(t_delta, 'JST')).strftime('%Y%m%d%H%M%S')
    st.download_button('Download spectrum CSV', df_spec.to_csv().encode('utf-8'),
                        f'fdtd_spectrum_{d}.csv', 'text/csv')

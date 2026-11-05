import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import time
import datetime
import functools
import plotly.graph_objects as go
from RCWA2D import rcwa_mh
import colour
from RCWA2D.structure_builder import (
    ENV_MATERIAL, SUBST_MATERIAL,
    columns_to_layers, get_layer_tuple,
    _expand_to_bands, _col_total_height,
    expand_graded_columns,
)
from RCWA2D.color_display import render_gsolver_style_display
from RCWA2D.gsolver_export import layers_to_gsolver_text, DEFAULT_NAME_MAP

try:
    import RCWA2D.rcwa_s4_backend as _s4b
    HAVE_S4 = _s4b.HAVE_S4
except ImportError:
    HAVE_S4 = False

st.set_page_config(
    page_title="Structural color simulator",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://koumyou.org/',
        'Report a bug': "https://koumyou.org/",
        'About': "# This is a structural color simulator app!"
    }
)

wl_min, wl_max, wl_n = 400.0, 800.0, 81
n_env = 1.0
inc_angle = 0.0
nk_idx_subst = 0
nk_idx_film = 0
norder = 11

def tictoc(func):
    def _wrapper(*args, **kwargs):
        t0 = time.time()
        r  = func(*args, **kwargs)
        print(f'time: {time.time()-t0:.6f} [sec]')
        return r
    return _wrapper


@functools.cache
def Rcwa1d_cached(pol, lambda0, kx0, period, layer, norder):
    ir, it = rcwa_mh.Rcwa1d(pol, lambda0, kx0, period, layer, norder)
    return (ir, it)


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


@st.cache_data
def get_nk_list():
    nk_list = []
    nk_dirs = "data//nk"
    files   = os.listdir(nk_dirs)
    for f in files:
        if os.path.isfile(os.path.join(nk_dirs, f)):
            nk_list.append(os.path.splitext(os.path.basename(f))[0])
    nk_list.sort()
    return nk_list


def make_nk_fn(nk_name_list=[]):
    nk_path = "data//nk//"
    fns = []
    for name in nk_name_list:
        if isinstance(name, (complex, float, int)):
            nk = complex(name)
            fns.append(lambda wl, _n=nk: _n)
        elif isinstance(name, str) and name.isnumeric():
            nk = complex(float(name))
            fns.append(lambda wl, _n=nk: _n)
        else:
            fpath = nk_path + name + '.nk'
            if os.path.isfile(fpath):
                d = np.loadtxt(fpath, comments=';', encoding='utf-8_sig')
                fn = interp1d(d[:, 0], d[:, 1] + d[:, 2]*1j, kind='linear', fill_value='extrapolate')
                fns.append(fn)
            else:
                try:    nk = complex(name)
                except: nk = complex(1.0)
                fns.append(lambda wl, _n=nk: _n)
    return fns


def build_nk_fn_map(material_names):
    seen, uniq = set(), []
    for n in material_names:
        if n in (ENV_MATERIAL, SUBST_MATERIAL) or n in seen:
            continue
        seen.add(n); uniq.append(n)
    return dict(zip(uniq, make_nk_fn(uniq)))


# ----------------------------------------------------------------
def create_structure_figure(layers, pitch_nm, nk_subst_name, col_widths=None):
    shapes, annotations = [], []
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
              '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    uniq_mat = sorted({m for L in layers for m, w in L['segments']
                       if m not in (ENV_MATERIAL, SUBST_MATERIAL)})
    color_map   = {m: colors[i % len(colors)] for i, m in enumerate(uniq_mat)}
    env_color   = 'rgba(240,240,240,0.95)'
    subst_color = 'rgba(160,200,230,0.9)'

    total_th  = sum(L['thickness'] for L in layers)
    subst_th  = total_th * 0.2 if total_th > 0 else 50

    shapes.append(go.layout.Shape(
        type="rect", xref="x", yref="y",
        x0=0, y0=-subst_th, x1=pitch_nm, y1=0,
        fillcolor="lightgrey", line_width=0
    ))
    annotations.append(dict(
        x=pitch_nm/2, y=-subst_th/2,
        text=nk_subst_name, showarrow=False, font=dict(color="black")
    ))

    current_y = 0.0
    for L in reversed(layers):
        d, cursor = L['thickness'], 0.0
        for mat, w in L['segments']:
            x0, x1 = pitch_nm * cursor, pitch_nm * (cursor + w)
            if   mat == ENV_MATERIAL:   clr = env_color
            elif mat == SUBST_MATERIAL: clr = subst_color
            else:                       clr = color_map.get(mat, '#cccccc')
            shapes.append(go.layout.Shape(
                type="rect", xref="x", yref="y",
                x0=x0, y0=current_y, x1=x1, y1=current_y + d,
                fillcolor=clr, line=dict(color="black", width=1)
            ))
            if mat not in (ENV_MATERIAL, SUBST_MATERIAL) and (x1-x0) > pitch_nm*0.04:
                annotations.append(dict(
                    x=(x0+x1)/2, y=current_y+d/2,
                    text=f"{mat}<br>{d:g}nm", showarrow=False,
                    font=dict(color="white", size=10)
                ))
            elif mat == SUBST_MATERIAL:
                annotations.append(dict(
                    x=(x0+x1)/2, y=current_y+d/2,
                    text=f"基板<br>{d:g}nm", showarrow=False,
                    font=dict(color="black", size=9)
                ))
            cursor += w
        current_y += d

    # 列境界を点線で表示
    if col_widths:
        tw = sum(col_widths)
        cx = 0.0
        for cw in col_widths[:-1]:
            cx += cw
            xp = pitch_nm * cx / tw
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y",
                x0=xp, y0=-subst_th, x1=xp, y1=current_y,
                line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot")
            ))

    fig = go.Figure()
    fig.update_layout(
        title="Film Stack Visualization",
        xaxis_title="Position (nm)", yaxis_title="Height (nm)",
        xaxis=dict(range=[0, pitch_nm], showgrid=False, zeroline=False),
        yaxis=dict(range=[-subst_th, current_y*1.1 if current_y>0 else 100],
                   scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False),
        shapes=shapes, annotations=annotations, plot_bgcolor='white'
    )
    return fig


def get_structure_breakdown_df(columns, flat_align=False):
    """各高さでの列材料を示す DataFrame を返す。"""
    align_h = max((c.get('subst_raise', 0.0) for c in columns), default=0.0) if flat_align else 0.0
    col_bands = [_expand_to_bands(col, flat_align_height=align_h) for col in columns]
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
        unique_mats = set()
        for ci, (col, bands) in enumerate(zip(columns, col_bands)):
            mat = ENV_MATERIAL
            for blo, bhi, bmat, *_ in bands:
                if blo <= h_mid < bhi:
                    mat = bmat; break
            label = '空気' if mat == ENV_MATERIAL else ('基板' if mat == SUBST_MATERIAL else mat)
            _tag = ', graded近似' if col.get('raise_mode') == 'graded' else ''
            row[f'C{ci+1}({col["width"]:.0f}nm{_tag})'] = label
            unique_mats.add(mat)
        n_uniq = len(unique_mats)
        row['層の種類'] = '均一層' if n_uniq == 1 else f'{n_uniq}材料 格子層'
        data.append(row)
    return pd.DataFrame(data)


def create_crosssection_fig(layers, pitch_nm, col_widths=None, n_repeats=2):
    """2D 周期構造断面図（複数周期表示）を生成する。"""
    shapes, annotations = [], []
    palette = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    uniq_mat = sorted({m for L in layers for m, w in L['segments']
                       if m not in (ENV_MATERIAL, SUBST_MATERIAL)})
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(uniq_mat)}
    env_color   = 'rgba(210,210,210,0.30)'
    subst_color = 'rgba(120,160,210,0.85)'
    subst_bg    = 'rgba(160,160,160,0.70)'

    total_h  = sum(L['thickness'] for L in layers)
    subst_sh = max(total_h * 0.10, 40.0)
    total_xw = pitch_nm * n_repeats

    for rep in range(n_repeats):
        xoff = pitch_nm * rep
        # 基板ブロック
        shapes.append(go.layout.Shape(
            type='rect', xref='x', yref='y',
            x0=xoff, y0=-subst_sh, x1=xoff + pitch_nm, y1=0,
            fillcolor=subst_bg, line=dict(color='grey', width=0.5)
        ))
        current_y = 0.0
        for L in reversed(layers):
            d = L['thickness']
            cursor = 0.0
            for mat, w in L['segments']:
                if w < 1e-9:
                    continue
                x0 = xoff + pitch_nm * cursor
                x1 = xoff + pitch_nm * (cursor + w)
                clr = (env_color   if mat == ENV_MATERIAL   else
                       subst_color if mat == SUBST_MATERIAL else
                       color_map.get(mat, '#cccccc'))
                shapes.append(go.layout.Shape(
                    type='rect', xref='x', yref='y',
                    x0=x0, y0=current_y, x1=x1, y1=current_y + d,
                    fillcolor=clr, line=dict(color='rgba(0,0,0,0.20)', width=0.5)
                ))
                # 材料名ラベル（最初の周期のみ、幅が十分なとき）
                if rep == 0 and mat not in (ENV_MATERIAL, SUBST_MATERIAL):
                    if (x1 - x0) > pitch_nm * 0.04 and d > 5:
                        annotations.append(dict(
                            x=(x0 + x1) / 2, y=current_y + d / 2,
                            text=mat, showarrow=False,
                            font=dict(size=8, color='white'), bgcolor='rgba(0,0,0,0)'
                        ))
                cursor += w
            current_y += d

        # 列境界点線 + 列番号ラベル
        if col_widths:
            tw = sum(col_widths)
            cx = 0.0
            for ci, cw in enumerate(col_widths):
                xp = xoff + pitch_nm * cx / tw
                if ci > 0:
                    shapes.append(go.layout.Shape(
                        type='line', xref='x', yref='y',
                        x0=xp, y0=-subst_sh, x1=xp, y1=total_h,
                        line=dict(color='rgba(0,0,0,0.45)', width=1, dash='dot')
                    ))
                if rep == 0:
                    annotations.append(dict(
                        x=xoff + pitch_nm * (cx + cw / 2) / tw,
                        y=-subst_sh * 0.65,
                        text=f'<b>C{ci+1}</b><br>{cw:.0f}',
                        showarrow=False, font=dict(size=9),
                        align='center'
                    ))
                cx += cw

        # 周期境界（実線）
        bx = xoff if rep > 0 else None
        for bxv in ([xoff] if rep > 0 else []):
            shapes.append(go.layout.Shape(
                type='line', xref='x', yref='y',
                x0=bxv, y0=-subst_sh, x1=bxv, y1=total_h,
                line=dict(color='black', width=1.5)
            ))

    # 右端境界線
    shapes.append(go.layout.Shape(
        type='line', xref='x', yref='y',
        x0=total_xw, y0=-subst_sh, x1=total_xw, y1=total_h,
        line=dict(color='black', width=1.5)
    ))

    # 凡例
    legend_shapes, legend_ann = [], []
    ly = total_h * 0.98
    lx0 = total_xw * 1.01
    lx1 = lx0 + pitch_nm * 0.06
    for mat, clr in color_map.items():
        legend_shapes.append(go.layout.Shape(
            type='rect', xref='x', yref='y',
            x0=lx0, y0=ly - total_h * 0.04, x1=lx1, y1=ly,
            fillcolor=clr, line_width=1
        ))
        legend_ann.append(dict(
            x=lx1 + pitch_nm * 0.01, y=ly - total_h * 0.02,
            text=mat, showarrow=False, xanchor='left',
            font=dict(size=10)
        ))
        ly -= total_h * 0.07

    fig = go.Figure()
    fig.update_layout(
        title=dict(text='周期構造断面図  ( x: 位置, y: 高さ )', font=dict(size=14)),
        shapes=shapes + legend_shapes,
        annotations=annotations + legend_ann,
        xaxis=dict(title='x 位置 [nm]', range=[-pitch_nm * 0.01, total_xw + pitch_nm * 0.30],
                   showgrid=False, zeroline=False),
        yaxis=dict(title='高さ [nm]', range=[-subst_sh * 1.3, total_h * 1.05],
                   showgrid=True, gridcolor='rgba(200,200,200,0.4)', zeroline=True,
                   zerolinecolor='rgba(0,0,0,0.4)', zerolinewidth=1.5),
        height=max(380, min(720, int(total_h * 0.55 + 120))),
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


@tictoc
def calc_rcwa1d_mh(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env_val, nk_subst_fn):
    nwl = len(wl_nm_ar)
    irp = np.empty([nwl, norder], dtype=float)
    itp = np.empty([nwl, norder], dtype=float)
    irs = np.empty([nwl, norder], dtype=float)
    its = np.empty([nwl, norder], dtype=float)
    for idx, wl_nm in enumerate(wl_nm_ar):
        wl_um = float(wl_nm / 1000.0)
        layer = get_layer_tuple(float(wl_nm), layers, nk_fn_map, n_env_val, nk_subst_fn)
        coef  = float(2*np.pi*np.sin(inc_angle_rad)/wl_um)
        irp[idx,:], itp[idx,:] = Rcwa1d_cached('p', wl_um, coef, pitch_um, layer, norder)
        irs[idx,:], its[idx,:] = Rcwa1d_cached('s', wl_um, coef, pitch_um, layer, norder)
    return (irp, itp, irs, its)


def _layers_cache_key(layers):
    """layers (list of dict with tuple-able contents) -> hashable key for st.cache_data."""
    return tuple(
        (L['thickness'], tuple((m, round(w, 12)) for m, w in L['segments']))
        for L in layers
    )


@st.cache_data(show_spinner=False)
def _calc_rcwa1d_s4_cached(wl_nm_tuple, inc_angle_rad, pitch_um, norder, layers_key,
                            material_names, nk_samples, n_env_val, nk_subst_samples):
    """S4 is far slower per-point than rcwa_mh (no eigen-cache across wavelengths within one
    call, and layer/region setup cost), so results are cached on the exact (structure,
    wavelength grid, angle, norder) tuple -- reruns from unrelated UI tweaks (color picker,
    plot-order selection, etc.) hit this cache instead of resolving S4 again.
    material_names/nk_samples/nk_subst_samples pre-resolve the nk_fn_map/nk_subst_fn
    closures (which aren't hashable) into plain per-wavelength (re, im) float-pair tuples --
    Streamlit's hasher can hash plain floats/tuples but not `complex`, so the complex values
    are decomposed here and reconstructed inside.
    """
    import RCWA2D.rcwa_s4_backend as s4b
    layers = [{'thickness': th, 'segments': list(segs)} for th, segs in layers_key]
    nk_fn_map = {
        name: (lambda wl, _s=samples, _wl=wl_nm_tuple: complex(*_s[_wl.index(wl)]))
        for name, samples in zip(material_names, nk_samples)
    }
    nk_subst_fn = lambda wl, _s=nk_subst_samples, _wl=wl_nm_tuple: complex(*_s[_wl.index(wl)])
    wl_nm_ar = np.array(wl_nm_tuple, dtype=float)
    return s4b.calc_rcwa1d_s4(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers,
                               nk_fn_map, n_env_val, nk_subst_fn)


@tictoc
def calc_rcwa1d_s4(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env_val, nk_subst_fn):
    wl_nm_tuple = tuple(float(w) for w in wl_nm_ar)
    material_names = sorted(nk_fn_map.keys())
    nk_samples = tuple(
        tuple((c.real, c.imag) for c in (complex(nk_fn_map[m](wl)) for wl in wl_nm_tuple))
        for m in material_names
    )
    nk_subst_samples = tuple((c.real, c.imag) for c in (complex(nk_subst_fn(wl)) for wl in wl_nm_tuple))
    return _calc_rcwa1d_s4_cached(wl_nm_tuple, inc_angle_rad, pitch_um, norder,
                                   _layers_cache_key(layers), tuple(material_names),
                                   nk_samples, n_env_val, nk_subst_samples)


def calc_rcwa1d(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env_val, nk_subst_fn):
    fn = calc_rcwa1d_s4 if st.session_state.get('solver_backend') == 'S4' else calc_rcwa1d_mh
    return fn(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env_val, nk_subst_fn)


# ================================================================
# Streamlit UI
# ================================================================
st.title('Structural color simulator (v4: 列ベース構造ビルダー)')

st.sidebar.header('Light parameters')
nk_namelist = get_nk_list()
nk_idx_subst = nk_namelist.index('Silicon')
nk_idx_film  = nk_namelist.index('SiO2')

angle_mode = st.sidebar.radio('Angle mode', ('Single angle (2D)', 'Angle sweep (3D)'), key='angle_mode')
if angle_mode == 'Single angle (2D)':
    inc_angle = st.sidebar.number_input('Incident angle [deg]', 0.0, 89.0, 0.0, 0.1, format='%3.1f')
else:
    angle_range = st.sidebar.slider('Angle range [deg]', 0.0, 89.0, (0.0, 80.0), 1.0, format='%.0f')
    angle_step  = st.sidebar.number_input('Angle step [deg]', 0.5, 45.0, 2.0, 0.5, format='%.1f')
    inc_angle   = angle_range[0]

spMenu = ('Visible[380-780nm]','UV[200-400nm]','NIR[700-1000nm]','All[200-1000nm]','Any')
wl_option = st.sidebar.selectbox('Spectrum range', spMenu)
if   wl_option == spMenu[0]: wl_min, wl_max, wl_n = 380.0, 780.0, 81
elif wl_option == spMenu[1]: wl_min, wl_max, wl_n = 200.0, 400.0, 101
elif wl_option == spMenu[2]: wl_min, wl_max, wl_n = 700.0, 1000.0, 61
elif wl_option == spMenu[3]: wl_min, wl_max, wl_n = 200.0, 1000.0, 81
if wl_option == spMenu[4]:
    wl_range = st.sidebar.slider('Wavelength range [nm]', 200.0, 1000.0, (wl_min, wl_max), 20.0, format='%.0f')
    wl_min, wl_max = wl_range
    wl_n = st.sidebar.number_input('Number of Wavelength', 11, 101, int(wl_n), 10, format='%d', key='wln')

st.sidebar.header('Atmosphere')
n_env = st.sidebar.number_input('Refractive index (air:1.00)', 1.0, 3.0, 1.0, 0.01, format='%3.2f', key='nenv')
st.sidebar.header('Substrate')
nk_subst_name = st.sidebar.selectbox('Substrate', nk_namelist, index=nk_idx_subst, key='substrate')
_custom_subst = st.sidebar.checkbox(
    'カスタム n+ik を直接入力 (constant override)',
    value=False, key='custom_subst_nk',
    help='Gsolverなど他ツールの基板光学定数を直接指定して比較する場合に使用')
if _custom_subst:
    _sc1, _sc2 = st.sidebar.columns(2)
    with _sc1:
        _subst_n = st.number_input('基板 n', 0.0, 20.0,
            st.session_state.get('subst_n_custom', 3.0), 0.01, format='%5.3f', key='subst_n_custom')
    with _sc2:
        _subst_k = st.number_input('基板 k', 0.0, 20.0,
            st.session_state.get('subst_k_custom', 4.0), 0.01, format='%5.3f', key='subst_k_custom')
    st.sidebar.caption(f'使用値: n={_subst_n:.3f}, k={_subst_k:.3f} (定数, 全波長共通)')

# ---- header ----
st.header('Structural color of patterned film using 1D RCWA/FMM')
norder = st.number_input('nOrder', min_value=5, max_value=101, value=norder, step=2, format='%d', key='nOrder')

st.sidebar.header('Solver backend')
if HAVE_S4:
    _backend_label = st.sidebar.radio(
        'RCWA backend', ['rcwa_mh (高速)', 'S4'], horizontal=True,
        help='rcwa_mh: 自製C++実装。高速でキャッシュも効く。\n'
             'S4: Stanford Stratified Structure Solver。独立実装による検証・比較用。'
             '波長数×角度数が多いと遅くなるので、確認用途向け。',
    )
    st.session_state.solver_backend = 'S4' if _backend_label == 'S4' else 'rcwa_mh'
else:
    st.session_state.solver_backend = 'rcwa_mh'
    st.sidebar.caption('S4 未インストールのため rcwa_mh のみ使用可能')

# ================================================================
# Column builder
# ================================================================
st.subheader('Structure — Column builder')
st.caption(
    '周期を「列」に分割し、各列で **基板の上げ高さ** と **フィルム層** を独立に設定します。'
    '列の幅の合計が自動的にピッチになります。列は左から右の順序です。'
)

if 'columns' not in st.session_state:
    st.session_state.columns     = []
    st.session_state._ncol_id    = 0
    st.session_state._nlayer_id  = 0

def _new_col_id():
    st.session_state._ncol_id += 1
    return st.session_state._ncol_id

def _new_layer_id():
    st.session_state._nlayer_id += 1
    return st.session_state._nlayer_id

if st.button('＋ 列を追加 (Add column)'):
    st.session_state.columns.append({
        'id': _new_col_id(), 'width': 300.0,
        'subst_raise': 0.0, 'raise_mode': 'sharp', 'film_layers': [],
    })

_col_rm = _col_mv = _lrm = _ladd = None
# _ladd は (col_index, layer_type) のタプル

for ci, col in enumerate(st.session_state.columns):
    cid     = col['id']
    total_h = _col_total_height(col)
    label   = f"Column {ci+1}  │  幅 {col['width']:.0f} nm  │  高さ {total_h:.0f} nm"
    with st.expander(label, expanded=True):

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            col['width'] = st.number_input(
                '幅 Width [nm]', 1.0, 1e6, col['width'], 1.0, format='%g', key=f'cw_{cid}')
        with cc2:
            col['subst_raise'] = st.number_input(
                '基板上げ高さ [nm]', 0.0, 1e6, col['subst_raise'], 1.0, format='%g', key=f'csr_{cid}')
        with cc3:
            _raise_opts = ['sharp (段差)', 'graded (なだらか)']
            _raise_idx  = 1 if col.get('raise_mode', 'sharp') == 'graded' else 0
            _raise_sel  = st.selectbox(
                '上げ形状', _raise_opts, index=_raise_idx, key=f'crmode_{cid}',
                disabled=(col['subst_raise'] <= 0),
                help='graded: 実際のエッチング/リソグラフィ断面に近い、肩の丸まった'
                     'なだらかなプロファイル(super-Gaussian)。段差の代わりに'
                     '列端でなだらかに隣列へ接続します。')
            col['raise_mode'] = 'graded' if _raise_sel.startswith('graded') else 'sharp'

        if col['subst_raise'] > 0:
            if col.get('raise_mode') == 'graded':
                _n = 0.02 * col['width']
                st.caption(
                    f'▸ {nk_subst_name}（基板材質）: なだらか上げ（次数 n={_n:.2f}）。'
                    f'ピーク {col["subst_raise"]:.0f} nm、列端では '
                    f'≈{col["subst_raise"] / np.e:.0f} nm まで下がり、隣列へなだらかに連続します。'
                )
            else:
                st.caption(f'▸ {nk_subst_name}（基板材質）: 0 → {col["subst_raise"]:.0f} nm')

        st.markdown('**フィルム層 Film layers — 下から上 (bottom → top)**')

        for li, fl in enumerate(col['film_layers']):
            lid   = fl['id']
            ftype = fl.get('type', 'simple')

            # ---- 単層 ----
            if ftype == 'simple':
                lc1, lc2, lc3 = st.columns([3, 2, 1])
                with lc1:
                    idx_ = nk_namelist.index(fl['material']) if fl['material'] in nk_namelist else nk_idx_film
                    fl['material'] = st.selectbox(f'材質 #{li+1}', nk_namelist, index=idx_, key=f'lm_{cid}_{lid}')
                with lc2:
                    fl['thickness'] = st.number_input(
                        f'厚さ #{li+1} [nm]', 0.0, 1e6, fl.get('thickness', 100.0), 1.0, format='%g', key=f'lt_{cid}_{lid}')
                with lc3:
                    st.write('')
                    if st.button('✕', key=f'lrm_{cid}_{lid}', help='この層を削除'):
                        _lrm = (ci, li)

            # ---- ペア繰り返し ----
            elif ftype == 'pair':
                st.markdown(f'*🔄 ペア繰り返し層 #{li+1}*')
                st.caption('A = 上側（空気側）/ B = 下側（基板側） ← ppap.py の "Layer A (top)" と同一規約')
                p1, p2, p3 = st.columns([2, 2, 1])
                with p1:
                    ia_ = nk_namelist.index(fl.get('mat_a', nk_namelist[nk_idx_film])) if fl.get('mat_a','') in nk_namelist else nk_idx_film
                    fl['mat_a'] = st.selectbox(f'材質A 上側 #{li+1}', nk_namelist, index=ia_, key=f'pma_{cid}_{lid}')
                    fl['th_a']  = st.number_input(f'厚さA #{li+1} [nm]', 0.0, 1e6, fl.get('th_a', 50.0), 1.0, format='%g', key=f'pta_{cid}_{lid}')
                with p2:
                    ib_ = nk_namelist.index(fl.get('mat_b', nk_namelist[0])) if fl.get('mat_b','') in nk_namelist else 0
                    fl['mat_b'] = st.selectbox(f'材質B 下側 #{li+1}', nk_namelist, index=ib_, key=f'pmb_{cid}_{lid}')
                    fl['th_b']  = st.number_input(f'厚さB #{li+1} [nm]', 0.0, 1e6, fl.get('th_b', 50.0), 1.0, format='%g', key=f'ptb_{cid}_{lid}')
                with p3:
                    fl['n_pairs'] = st.number_input(f'繰り返し #{li+1}', 1, 200, fl.get('n_pairs', 3), 1, format='%d', key=f'pn_{cid}_{lid}')
                    pair_total = fl['n_pairs'] * (fl['th_a'] + fl['th_b'])
                    st.caption(f'合計 {pair_total:.0f} nm')
                    if st.button('✕', key=f'lrm_{cid}_{lid}', help='この層を削除'):
                        _lrm = (ci, li)

            # ---- 台形 ----
            elif ftype == 'taper':
                st.markdown(f'*📐 台形層 #{li+1}*')
                t1, t2, t3 = st.columns([2, 2, 2])
                with t1:
                    it_ = nk_namelist.index(fl.get('material', nk_namelist[nk_idx_film])) if fl.get('material','') in nk_namelist else nk_idx_film
                    fl['material']  = st.selectbox(f'材質 #{li+1}', nk_namelist, index=it_, key=f'tm_{cid}_{lid}')
                    fl['thickness'] = st.number_input(f'総厚さ #{li+1} [nm]', 0.0, 1e6, fl.get('thickness', 200.0), 1.0, format='%g', key=f'tth_{cid}_{lid}')
                with t2:
                    fb_pct = fl.get('frac_bottom', 1.0) * 100.0
                    ft_pct = fl.get('frac_top',    0.5) * 100.0
                    fl['frac_bottom'] = st.number_input(f'幅割合(下) #{li+1} [%]', 0.0, 100.0, fb_pct, 1.0, format='%g', key=f'tfb_{cid}_{lid}') / 100.0
                    fl['frac_top']    = st.number_input(f'幅割合(上) #{li+1} [%]', 0.0, 100.0, ft_pct, 1.0, format='%g', key=f'tft_{cid}_{lid}') / 100.0
                with t3:
                    fl['n_slices']  = st.number_input(f'スライス数 #{li+1}', 2, 100, fl.get('n_slices', 10), 1, format='%d', key=f'tns_{cid}_{lid}')
                    align_opts      = ['Center', 'Left', 'Right']
                    ai_             = align_opts.index(fl.get('alignment', 'Center'))
                    fl['alignment'] = st.selectbox(f'配置 #{li+1}', align_opts, index=ai_, key=f'tal_{cid}_{lid}')
                    if st.button('✕', key=f'lrm_{cid}_{lid}', help='この層を削除'):
                        _lrm = (ci, li)

            st.divider()

        # 層追加ボタン (3種)
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button('＋ 単層', key=f'ladd_{cid}', help='単一材料の均一層を追加'):
                _ladd = (ci, 'simple')
        with bc2:
            if st.button('＋ ペア繰り返し', key=f'ladd_p_{cid}', help='A/B 2材料を N 回繰り返す多層を追加'):
                _ladd = (ci, 'pair')
        with bc3:
            if st.button('＋ 台形', key=f'ladd_t_{cid}', help='列内の幅が下から上へ線形変化する台形層を追加'):
                _ladd = (ci, 'taper')

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            if st.button('◀ 左へ', key=f'cl_{cid}', disabled=(ci == 0)):
                _col_mv = (ci, -1)
        with mc2:
            if st.button('▶ 右へ', key=f'cr_{cid}', disabled=(ci == len(st.session_state.columns)-1)):
                _col_mv = (ci, 1)
        with mc3:
            if st.button('✕ この列を削除', key=f'crm_{cid}'):
                _col_rm = ci

if _lrm is not None:
    ci, li = _lrm
    st.session_state.columns[ci]['film_layers'].pop(li)
    st.rerun()
if _ladd is not None:
    ci, ltype = _ladd
    if ltype == 'simple':
        st.session_state.columns[ci]['film_layers'].append({
            'id': _new_layer_id(), 'type': 'simple',
            'material': nk_namelist[nk_idx_film], 'thickness': 100.0
        })
    elif ltype == 'pair':
        st.session_state.columns[ci]['film_layers'].append({
            'id': _new_layer_id(), 'type': 'pair',
            'mat_a': nk_namelist[nk_idx_film], 'th_a': 50.0,
            'mat_b': nk_namelist[nk_idx_subst], 'th_b': 50.0,
            'n_pairs': 3
        })
    elif ltype == 'taper':
        st.session_state.columns[ci]['film_layers'].append({
            'id': _new_layer_id(), 'type': 'taper',
            'material': nk_namelist[nk_idx_film], 'thickness': 200.0,
            'frac_bottom': 1.0, 'frac_top': 0.5,
            'n_slices': 10, 'alignment': 'Center'
        })
    st.rerun()
if _col_rm is not None:
    st.session_state.columns.pop(_col_rm)
    st.rerun()
if _col_mv is not None:
    ci, d = _col_mv; j = ci + d
    st.session_state.columns[ci], st.session_state.columns[j] = \
        st.session_state.columns[j], st.session_state.columns[ci]
    st.rerun()

# ---- Convert to RCWA layers ----
col_widths = [c['width'] for c in st.session_state.columns]
pitch_nm   = sum(col_widths) if col_widths else 500.0
pitch_um   = pitch_nm / 1000.0

_has_graded = any(c.get('raise_mode') == 'graded' for c in st.session_state.columns)
if _has_graded:
    graded_n_slices = st.number_input(
        'なだらか基板プロファイルの分割数 (列ごと)', 2, 100, 10, 1, format='%d',
        key='graded_n_slices',
        help='graded設定の列を、この本数だけ細い子列に分割してsuper-Gaussian'
             'プロファイル（および隣接列への裾の侵入）を階段状に近似します。'
             '多いほど滑らかですが、RCWA解析層数がおおむね比例して増え、'
             '特にS4バックエンドでは計算が大きく遅くなります。')
    layers = columns_to_layers(expand_graded_columns(st.session_state.columns, int(graded_n_slices)))
else:
    layers = columns_to_layers(st.session_state.columns)

if st.session_state.columns:
    st.caption(f'ピッチ = {pitch_nm:.1f} nm  ／  RCWA 解析層数 = {len(layers)} 層  ／  モデル: 等形成膜（列ごとに膜を積む）')
else:
    st.info('「列を追加」ボタンで列を追加してください（空の場合は平坦基板として計算されます）。')

# ================================================================
# Gsolver export
# ================================================================
with st.expander('📤 Gsolverインポート用テキストを生成', expanded=False):
    st.caption(
        '現在の構造（列・膜構成）を GSolver の File > Import Text 用テキストに変換します。'
        '材質名は GSolver.ini の [CONSTANTS] カタログに同名で登録されている必要があります'
        f'（基板は現在選択中の「{nk_subst_name}」という名前でカタログに登録されている前提。'
        'カスタムn+ik入力を使っている場合、その値はこのエクスポートには反映されません — '
        '先にGSolver.iniへ該当のn+ikを登録し、名前を揃えてください）。'
        'インポート後は必ずGSolver側の断面プレビューで意図した形状か目視確認し、'
        'Approximateボタンは押さないこと（寸法がCanvas Grid精度に再量子化されます）。'
        'ORDERSは43以上を推奨します（詳細はHANDOVER.md参照）。'
    )
    if layers:
        try:
            _name_map = dict(DEFAULT_NAME_MAP)
            _name_map[SUBST_MATERIAL] = nk_subst_name
            gsolver_text = layers_to_gsolver_text(layers, name_map=_name_map)
            st.text_area('プレビュー', gsolver_text, height=200, key='gsolver_export_preview')
            st.download_button(
                'Gsolverインポート用テキストをダウンロード (.txt)',
                gsolver_text,
                f'gsolver_import_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt',
                'text/plain')
        except ValueError as e:
            st.error(f'エクスポートに失敗しました: {e}')
    else:
        st.info('列を追加すると、ここにGsolverインポート用テキストが表示されます。')

# ================================================================
# Film Stack Visualization
# ================================================================
st.subheader('Film Stack Visualization')
fig_structure = create_structure_figure(layers, pitch_nm, nk_subst_name, col_widths=col_widths)
st.plotly_chart(fig_structure, use_container_width=True)

# ================================================================
# 周期構造断面ビューア
# ================================================================
with st.expander('🔬 周期構造断面ビューア / 層テーブル', expanded=False):
    st.caption(
        '横軸 = x 位置 [nm]（1周期 = ピッチ）、縦軸 = 高さ [nm]。'
        '同じユニットセルが横方向に繰り返す周期構造を示します。'
        '等形成膜モードでは高さによって列ごとに異なる材料が現れます。'
    )
    _rep_c1, _rep_c2 = st.columns([4, 1])
    with _rep_c2:
        _n_rep = st.number_input('表示周期数', 1, 6, 2, 1, key='crosssec_nrep')
    if layers:
        fig_cs = create_crosssection_fig(layers, pitch_nm,
                                         col_widths=col_widths, n_repeats=int(_n_rep))
        st.plotly_chart(fig_cs, use_container_width=True)
    else:
        st.info('列を追加すると断面図が表示されます。')

    if st.session_state.columns:
        st.markdown('**層テーブル — 各高さでの材料と層の種類**')
        df_struct = get_structure_breakdown_df(st.session_state.columns)
        # 格子層を強調表示
        def _highlight_lattice(row):
            if '格子層' in str(row.get('層の種類', '')):
                return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)
        st.dataframe(
            df_struct.style.apply(_highlight_lattice, axis=1),
            use_container_width=True, hide_index=True, height=320
        )

# ================================================================
# Spectrum / RCWA
# ================================================================
st.subheader('Spectrum')

material_names = []
for _col in st.session_state.columns:
    for _fl in _col['film_layers']:
        _ft = _fl.get('type', 'simple')
        if   _ft == 'simple': material_names.append(_fl.get('material', ''))
        elif _ft == 'pair':   material_names += [_fl.get('mat_a',''), _fl.get('mat_b','')]
        elif _ft == 'taper':  material_names.append(_fl.get('material', ''))
material_names = [m for m in material_names if m and m not in (ENV_MATERIAL, SUBST_MATERIAL)]
nk_fn_map   = build_nk_fn_map(material_names)
if st.session_state.get('custom_subst_nk', False):
    _cn = st.session_state.get('subst_n_custom', 3.0)
    _ck = st.session_state.get('subst_k_custom', 4.0)
    nk_subst_fn = make_nk_fn([complex(_cn, _ck)])[0]
else:
    nk_subst_fn = make_nk_fn([nk_subst_name])[0]

wl_nm_ar   = np.linspace(wl_min, wl_max, wl_n, dtype=float)
idex_0     = norder // 2
order_list = list(range(-idex_0, idex_0 + 1))

def _order_label(m): return '0 (specular)' if m == 0 else f'{m:+d}'
def _idx_for_order(m): return idex_0 + m

def _sd_to_hex(arr, name):
    sd = colour.SpectralDistribution(arr, name=name)
    sd.wavelengths = wl_nm_ar
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    ill  = colour.SDS_ILLUMINANTS['D65']
    XYZ  = colour.sd_to_XYZ(sd, cmfs, ill)
    RGB  = colour.XYZ_to_sRGB(XYZ / 100)
    b    = [int(np.clip(round(v*255), 0, 255)) for v in RGB]
    return XYZ, '#'+format(b[0],'02x')+format(b[1],'02x')+format(b[2],'02x')

ocol1, ocol2 = st.columns(2)
with ocol1:
    plot_orders = st.multiselect(
        'Orders to show in spectrum plot', order_list, default=[0],
        format_func=_order_label, key='plot_orders')
    if not plot_orders: plot_orders = [0]
with ocol2:
    repr_order = st.selectbox(
        'Order for colorimetry / CSV / 3D surface', order_list,
        index=order_list.index(0), format_func=_order_label, key='repr_order')

_nonzero_orders = [m for m in order_list if m != 0]
_nonzero_idx    = [_idx_for_order(m) for m in _nonzero_orders]

# ---- 計算コントロール（画面右下に固定表示。RCWA計算は「計算」ボタンを押した時のみ実行し、
# 構造やパラメータのウィジェット操作だけでは自動再計算しない） ----
st.html("""
<style>
.st-key-calc_action_bar {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
    padding: 10px 14px 4px 14px;
    border-radius: 12px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.35);
    background-color: #ffffff;
    border: 1px solid rgba(0,0,0,0.15);
    width: 260px;
}
@media (prefers-color-scheme: dark) {
    .st-key-calc_action_bar {
        background-color: #262730;
        border-color: rgba(255,255,255,0.15);
    }
}
</style>
""")
with st.container(key='calc_action_bar'):
    _bcol1, _bcol2 = st.columns(2)
    with _bcol1:
        calc_clicked = st.button(
            '▶ 計算', key='calc_btn', type='primary', use_container_width=True,
            help='現在の構造・パラメータでRCWA計算を実行します。')
    with _bcol2:
        stop_clicked = st.button(
            '⏹ 停止', key='stop_btn', use_container_width=True,
            help='計算中にクリックすると、その時点(次の角度に進む前)で計算を打ち切ります。'
                 '直前に完了した計算結果があればそのまま表示され続けます。')
    _status_slot = st.empty()

def _calc_signature(mode):
    base = (
        st.session_state.get('solver_backend'),
        _layers_cache_key(layers),
        pitch_um, norder, n_env,
        nk_subst_name, bool(st.session_state.get('custom_subst_nk', False)),
        st.session_state.get('subst_n_custom'), st.session_state.get('subst_k_custom'),
        wl_option, wl_min, wl_max, wl_n,
    )
    if mode == 'single':
        return ('single', base, inc_angle)
    else:
        return ('sweep', base, tuple(angle_range), angle_step, repr_order)


def _run_calculation():
    """重いRCWA計算本体。「計算」ボタンが押された時のみ呼ばれる。"""
    if angle_mode == 'Single angle (2D)':
        inc_angle_rad = inc_angle * np.pi / 180.0
        irp, itp, irs, its = calc_rcwa1d(
            wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn)
        return {'mode': 'single', 'inc_angle': inc_angle,
                'irp': irp, 'itp': itp, 'irs': irs, 'its': its}

    angle_ar = np.arange(angle_range[0], angle_range[1] + angle_step/2, angle_step, dtype=float)
    nang = len(angle_ar)

    Rp2d        = np.empty([nang, wl_n], dtype=float)
    Rs2d        = np.empty([nang, wl_n], dtype=float)
    Tp2d        = np.empty([nang, wl_n], dtype=float)
    Ts2d        = np.empty([nang, wl_n], dtype=float)
    Rp_diff_2d  = np.zeros([nang, wl_n], dtype=float)
    Rs_diff_2d  = np.zeros([nang, wl_n], dtype=float)
    Rp_total_2d = np.zeros([nang, wl_n], dtype=float)
    Rs_total_2d = np.zeros([nang, wl_n], dtype=float)

    idx_repr = _idx_for_order(repr_order)
    prog = st.progress(0.0, text='Calculating...')
    for ia, ang in enumerate(angle_ar):
        ang_rad = ang * np.pi / 180.0
        irp, itp, irs, its = calc_rcwa1d(
            wl_nm_ar, ang_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn)
        Rp2d[ia, :] = irp[:, idx_repr]
        Rs2d[ia, :] = irs[:, idx_repr]
        Tp2d[ia, :] = itp[:, idx_repr]
        Ts2d[ia, :] = its[:, idx_repr]
        if _nonzero_idx:
            Rp_diff_2d[ia, :] = np.clip(irp[:, _nonzero_idx].sum(axis=1), 0, 1)
            Rs_diff_2d[ia, :] = np.clip(irs[:, _nonzero_idx].sum(axis=1), 0, 1)
        Rp_total_2d[ia, :] = np.clip(irp.sum(axis=1), 0, 1)
        Rs_total_2d[ia, :] = np.clip(irs.sum(axis=1), 0, 1)
        # st.progress() を呼ぶたびにStreamlitが「停止」クリック等の新しい操作の有無を
        # チェックする -- 新しい操作があればこの関数は途中で強制的に打ち切られ、
        # session_state.calc_cache は更新されない(=直前の結果がそのまま残る)。
        prog.progress((ia+1)/nang, text=f'Angle sweep {ia+1}/{nang}')
    prog.empty()

    swatch_thetas = [0.0, 30.0, 60.0]
    swatch = {k: [] for k in ('Rp_single', 'Rs_single', 'Rp_diff', 'Rs_diff', 'Rp_total', 'Rs_total')}
    show_swatches = (wl_option == spMenu[0])
    if show_swatches:
        for swatch_angle in swatch_thetas:
            swatch_angle_rad = swatch_angle * np.pi / 180.0
            s_irp, s_itp, s_irs, s_its = calc_rcwa1d(
                wl_nm_ar, swatch_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn)
            swatch['Rp_single'].append(s_irp[:, idx_repr].tolist())
            swatch['Rs_single'].append(s_irs[:, idx_repr].tolist())
            swatch['Rp_diff'].append((np.clip(s_irp[:, _nonzero_idx].sum(axis=1), 0, 1)
                                       if _nonzero_idx else np.zeros(wl_n)).tolist())
            swatch['Rs_diff'].append((np.clip(s_irs[:, _nonzero_idx].sum(axis=1), 0, 1)
                                       if _nonzero_idx else np.zeros(wl_n)).tolist())
            swatch['Rp_total'].append(np.clip(s_irp.sum(axis=1), 0, 1).tolist())
            swatch['Rs_total'].append(np.clip(s_irs.sum(axis=1), 0, 1).tolist())

    return {
        'mode': 'sweep', 'angle_ar': angle_ar, 'repr_order': repr_order,
        'show_swatches': show_swatches,
        'Rp2d': Rp2d, 'Rs2d': Rs2d, 'Tp2d': Tp2d, 'Ts2d': Ts2d,
        'Rp_diff_2d': Rp_diff_2d, 'Rs_diff_2d': Rs_diff_2d,
        'Rp_total_2d': Rp_total_2d, 'Rs_total_2d': Rs_total_2d,
        'swatch_thetas': swatch_thetas, 'swatch': swatch,
    }


_mode_key    = 'single' if angle_mode == 'Single angle (2D)' else 'sweep'
current_sig  = _calc_signature(_mode_key)

if calc_clicked:
    with st.spinner('計算中...'):
        _result = _run_calculation()
    st.session_state.calc_cache = {
        'signature': current_sig,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'wl_nm_ar': wl_nm_ar,
        'result': _result,
    }
elif stop_clicked and st.session_state.get('calc_cache') is not None:
    st.info('⏹ 計算を停止しました。直前の計算結果を表示しています。')

cache = st.session_state.get('calc_cache')
if cache is None:
    _status_slot.caption('まだ計算していません')
    st.info('👉 右下の「▶ 計算」ボタンを押して、現在の構造・パラメータでRCWA計算を実行してください。')
    st.stop()
_status_slot.caption(f'最終計算: {cache["timestamp"]}')

is_stale = (cache['signature'] != current_sig)
if is_stale:
    st.warning(
        f'⚠ 表示中の結果は前回計算時（{cache["timestamp"]}）の構造・パラメータに基づきます。'
        '変更を反映するには右下の「▶ 計算」を押して再計算してください。'
    )

# 以降はキャッシュされた計算結果のみを参照する（ライブの wl_nm_ar もキャッシュ時点の
# ものに揃え、stale時でも波長軸とスペクトル配列の長さが食い違わないようにする）。
wl_nm_ar = cache['wl_nm_ar']
result   = cache['result']

# ---- Single angle ----
if result['mode'] == 'single':
    irp, itp, irs, its = result['irp'], result['itp'], result['irs'], result['its']
    inc_angle = result['inc_angle']

    Rp = irp[:, _idx_for_order(repr_order)]
    Rs = irs[:, _idx_for_order(repr_order)]
    Tp = itp[:, _idx_for_order(repr_order)]
    Ts = its[:, _idx_for_order(repr_order)]

    Rp_diff = np.clip(sum(irp[:, i] for i in _nonzero_idx), 0, 1) if _nonzero_idx else np.zeros(len(wl_nm_ar))
    Rs_diff = np.clip(sum(irs[:, i] for i in _nonzero_idx), 0, 1) if _nonzero_idx else np.zeros(len(wl_nm_ar))

    fig = go.Figure()
    for m in sorted(plot_orders):
        idx_m = _idx_for_order(m)
        fig.add_trace(go.Scatter(x=wl_nm_ar, y=irp[:, idx_m], name=f'Rp (m={_order_label(m)})', mode='lines'))
        fig.add_trace(go.Scatter(x=wl_nm_ar, y=irs[:, idx_m], name=f'Rs (m={_order_label(m)})', mode='lines'))
    fig.update_layout(title=f'AOI {round(inc_angle,1)} deg', yaxis_zeroline=True, xaxis_zeroline=True)
    fig.update_xaxes(title_text='Wavelength(nm)')
    if plot_orders == [0]:
        fig.update_yaxes(title_text='R', range=[0, 1])
    else:
        all_y = np.concatenate([irp[:, _idx_for_order(m)] for m in plot_orders] +
                               [irs[:, _idx_for_order(m)] for m in plot_orders])
        y_top = max(float(all_y.max()) * 1.15, 1e-3)
        fig.update_yaxes(title_text='R', range=[0, min(y_top, 1.0)])
        if y_top < 0.1:
            st.info(f'高次回折光の最大反射率: {all_y.max():.4f}')
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f'色計算・CSV は m={_order_label(repr_order)} に基づきます。')

# ---- Angle sweep ----
else:
    angle_ar    = result['angle_ar']
    Rp2d        = result['Rp2d'];        Rs2d        = result['Rs2d']
    Tp2d        = result['Tp2d'];        Ts2d        = result['Ts2d']
    Rp_diff_2d  = result['Rp_diff_2d'];  Rs_diff_2d  = result['Rs_diff_2d']
    Rp_total_2d = result['Rp_total_2d']; Rs_total_2d = result['Rs_total_2d']
    swatch_thetas = result['swatch_thetas']; swatch = result['swatch']
    show_swatches = result['show_swatches']

    # ---- Unified color display (swatches + angle chart + spectrum chart), pixel-matched
    # to the standalone reference tool "Gsolver'S COLOR.html" -- see RCWA2D/color_display.py.
    disp_pol = st.radio('偏光 / Polarization', ('Rp', 'Rs'), horizontal=True, key='disp_pol')
    _repr_label = _order_label(result['repr_order'])
    _MODE_SINGLE = f'指定次数 (m={_repr_label})'
    _MODE_DIFF   = '0次光以外 (回折光 m≠0 合算)'
    _MODE_TOTAL  = '全次数合算 (0次光含む)'
    disp_order_mode = st.radio(
        '表示する光成分', (_MODE_SINGLE, _MODE_DIFF, _MODE_TOTAL),
        index=2, horizontal=True, key='disp_order_mode',
        help='上のOrder選択とは独立に、スウォッチ・角度チャート・スペクトルチャート'
             '全てに表示する光成分を切り替えます。「0次光以外」は m≠0 の全次数、'
             '「全次数合算」は0次光を含む全次数を合算した反射率 (0〜1にクリップ) を'
             '表示します。ピッチが波長より十分小さいと高次光は伝搬せず、'
             '0次光以外は常に0(黒)になります。')

    if disp_order_mode == _MODE_DIFF:
        Z_disp = Rp_diff_2d if disp_pol == 'Rp' else Rs_diff_2d
    elif disp_order_mode == _MODE_TOTAL:
        Z_disp = Rp_total_2d if disp_pol == 'Rp' else Rs_total_2d
    else:
        Z_disp = Rp2d if disp_pol == 'Rp' else Rs2d

    swatch_values = []
    if show_swatches:
        _swatch_pol_key = 'Rp' if disp_pol == 'Rp' else 'Rs'
        if disp_order_mode == _MODE_DIFF:
            swatch_values = swatch[f'{_swatch_pol_key}_diff']
        elif disp_order_mode == _MODE_TOTAL:
            swatch_values = swatch[f'{_swatch_pol_key}_total']
        else:
            swatch_values = swatch[f'{_swatch_pol_key}_single']

    render_gsolver_style_display(
        lambdas=wl_nm_ar, thetas=angle_ar, values=Z_disp,
        show_swatches=show_swatches,
        swatch_thetas=swatch_thetas, swatch_values=swatch_values,
    )

    inc_angle = float(angle_ar[0])
    Rp = Rp2d[0,:]; Rs = Rs2d[0,:]; Tp = Tp2d[0,:]; Ts = Ts2d[0,:]
    Rp_diff = Rp_diff_2d[0,:]; Rs_diff = Rs_diff_2d[0,:]
    st.caption(f'色計算・CSV は代表角度 {round(inc_angle,1)}°, m={_repr_label} に基づきます。')

# ================================================================
# Colorimetry
# ================================================================
if wl_option == spMenu[0]:
    st.subheader('Colorimetry')
    if repr_order != 0:
        st.warning('高次回折光は強度が小さいため色が黒に近くなります。')

    XYZ_Rp,      hRp      = _sd_to_hex(Rp,      'Rp')
    XYZ_Rs,      hRs      = _sd_to_hex(Rs,       'Rs')
    XYZ_Rp_diff, hRp_diff = _sd_to_hex(Rp_diff,  'Rp_diff')
    XYZ_Rs_diff, hRs_diff = _sd_to_hex(Rs_diff,  'Rs_diff')

    st.caption('回折光の色 = m≠0 全次数合算。ピッチ < 波長の場合は黒（高次光が伝搬しない）。')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.color_picker(f'Rp (m={_order_label(repr_order)})', hRp, key='cpRp'+hRp)
        st.write('XYZ', XYZ_Rp)
    with col2:
        st.color_picker(f'Rs (m={_order_label(repr_order)})', hRs, key='cpRs'+hRs)
        st.write('XYZ', XYZ_Rs)
    with col3:
        st.color_picker('Rp 回折光 (m≠0)', hRp_diff, key='cpRpd'+hRp_diff)
        st.write('XYZ', XYZ_Rp_diff)
    with col4:
        st.color_picker('Rs 回折光 (m≠0)', hRs_diff, key='cpRsd'+hRs_diff)
        st.write('XYZ', XYZ_Rs_diff)

# ================================================================
# CSV Download
# ================================================================
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
d   = datetime.datetime.now(JST).strftime('%Y%m%d%H%M%S')

st.subheader('Download')

df_spec = pd.DataFrame(np.stack([Rp, Rs], axis=1), index=wl_nm_ar, columns=['Rp','Rs'])
df_spec.index.name = 'Wavelength(nm)'
st.download_button('Download reflectance spectrum CSV (Rp, Rs)',
                   convert_df(df_spec), f'spectrum_{d}.csv', 'text/csv')

if result['mode'] == 'sweep':
    if float(angle_ar[0]) == 0.0:
        _xc = np.concatenate([-angle_ar[1:][::-1], angle_ar])
        def _mir(c): return np.concatenate([c[1:][::-1], c])
    else:
        _xc = np.concatenate([-angle_ar[::-1], angle_ar])
        def _mir(c): return np.concatenate([c[::-1], c])

    wl_cols = [f'{int(round(w))}nm' for w in wl_nm_ar]
    dfp = pd.DataFrame({c: _mir(Rp2d[:,i]) for i,c in enumerate(wl_cols)}, index=_xc)
    dfp.index.name = 'Angle(deg)'
    dfs = pd.DataFrame({c: _mir(Rs2d[:,i]) for i,c in enumerate(wl_cols)}, index=_xc)
    dfs.index.name = 'Angle(deg)'

    ac1, ac2 = st.columns(2)
    with ac1:
        st.download_button(f'Angle CSV Rp (m={_order_label(repr_order)})',
                           convert_df(dfp), f'angle_Rp_{d}.csv', 'text/csv')
    with ac2:
        st.download_button(f'Angle CSV Rs (m={_order_label(repr_order)})',
                           convert_df(dfs), f'angle_Rs_{d}.csv', 'text/csv')
    st.caption(f'行=角度(−{int(angle_ar[-1])}°～+{int(angle_ar[-1])}°)、列=各波長の反射率')

#from wsgiref.headers import tspecials
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

wl_min=400.0
wl_max=800.0
wl_n=81

n_env=1.0
inc_angle=0.0
nlayers=1

nk_idx_subst=0
nk_idx_film=0

pitch_nm = 500. # 周期（nm）
norder=11

ENV_MATERIAL   = '__ENV__'       # 媒質（環境屈折率）を表す特別なセグメント材質名
SUBST_MATERIAL = '__SUBSTRATE__' # 基板と同じ材質を参照する特別なセグメント材質名（ペデスタル層用）




def tictoc(func):
    def _wrapper(*args,**keywargs):
        start_time=time.time()
        result=func(*args,**keywargs)
        print('time: {:.9f} [sec]'.format(time.time()-start_time))
        return result
    return _wrapper


@functools.cache
def Rcwa1d_cached( pol, lambda0, kx0, period, layer, norder):
    ir, it = rcwa_mh.Rcwa1d( pol, lambda0, kx0, period, layer, norder)
    return (ir,it)


def order_n(i): return {1:"1st (Top)", 2:"2nd", 3:"3rd"}.get(i) or "%dth"%i



@st.cache_data
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

@st.cache_data
def get_nk_list():
    """
    フォルダ内のnkファイル名一覧の取得
    Parameters
    ----------
    nk_path : str
        nkファイルのパス.
    Returns
    -------
    name_list : list of str
        ファイル名のリスト

    """

    nk_list=[]
    nk_dirs="data//nk"
    files=os.listdir(nk_dirs)
    nk_files=[f for f in files if os.path.isfile(os.path.join(nk_dirs, f))]

    for nk_file in nk_files:
        basename = os.path.splitext(os.path.basename(nk_file))[0]
        nk_list.append(basename)

    if len(nk_list)<1:
        st.error('not find nk data in '+nk_dirs)
        files_data=glob.glob("data")
        st.error('dir of data =',files_data)
        files_nk=glob.glob("data\\nk")
        st.error('dir of data_nk =',files_nk)

    nk_list.sort()
    return nk_list

@st.cache_data
def calc_nk_list(nk_fn_list,wl):
    """
    各層の光学定数の関数リストと与えられた波長から、薄膜の光学定数リストを返す

    Parameters
    ----------
    nk_fn_list : list of fn(wl)
        光学定数の関数リスト.
    wl : float
        波長(nm).
    Returns
    -------
    nk_list : array of complex
        各層の光学定数.

    """
    nk_list=[]
    for nk in nk_fn_list:
        nk_list.append(nk(wl))
    return nk_list


def make_nk_fn(nk_name_list=[]):
    """
    各層の光学定数の関数を返す
    Parameters
    ----------
    nk_name_list : list of string
        光学定数名のリスト.

    Returns
    -------
    nk_fn_list : list of fn(wl)
        各層の光学定数の関数リスト.

    """
    nk_path="data//nk//" # nkファイルのパス
    nk_fn_list=[]
    for idx,nk_name in enumerate(nk_name_list):
        if isinstance(nk_name,complex) or isinstance(nk_name,float) or isinstance(nk_name,int):
            nk=complex(nk_name)
            nk_fn = lambda wavelength: nk
            #print(f'Idx={idx},Instance==numeric, val={nk}')
        elif isinstance(nk_name,str) and str(nk_name).isnumeric():
            nk=float(nk_name)
            nk_fn = lambda wavelength: nk
            #print(f'Idx={idx},Instance==str, val={nk}')
        else:
            fname_path=nk_path+nk_name+'.nk'
            if os.path.isfile(fname_path):
                nk_mat=np.loadtxt(fname_path,comments=';',encoding="utf-8_sig")
                #st.write(nk_mat)
                w_mat=nk_mat[:,0]
                n_mat=np.array(nk_mat[:,1]+nk_mat[:,2]*1j)
                #nk_fn= interp1d(w_mat,n_mat, kind='quadratic', fill_value='extrapolate')
                nk_fn= interp1d(w_mat,n_mat, kind='linear', fill_value='extrapolate')
                #print(f'Idx={idx},Instance=={fname_path} exist')
            else:
                try:
                    nk=complex(nk_name)
                except ValueError:
                    nk=complex(1.0)
                #print(f'Idx={idx},Instance=={fname_path} not exist, nk={nk}')
                nk_fn = lambda wavelength: nk

        nk_fn_list.append(nk_fn)
    #print(nk_fn_list)
    return nk_fn_list


def build_nk_fn_map(material_names):
    """
    材質名のリスト（重複あり）から、重複を除いた {材質名: fn(wl)} の辞書を返す。
    多リッジ/テーパーブロックで同じ材質が何度も参照されても、nkファイルのロードは1回だけになる。
    """
    unique_names=[]
    seen=set()
    for name in material_names:
        # SUBST_MATERIAL / ENV_MATERIAL は nk_fn_map に不要（get_layer_tuple で個別解決）
        if name in (ENV_MATERIAL, SUBST_MATERIAL):
            continue
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    fn_list=make_nk_fn(unique_names)
    return dict(zip(unique_names,fn_list))


def build_manual_segments(features):
    """
    1層分のセグメント列を構築する。
    features: [{'material':str,'width':float,'gap_before':float}, ...]
        gap_before: 直前のセグメント右端（最初は x=0）からの媒質ギャップ幅（ピッチ規格化）
    戻り値: (segments, was_scaled)
        segments: [(material_or_ENV, width), ...]  幅の合計は1.0
        was_scaled: 合計が1.0を超えていたため比例縮小したかどうか
    """
    total=sum(f['gap_before']+f['width'] for f in features)
    scale=1.0
    was_scaled=False
    if total>1.0+1e-9:
        scale=1.0/total
        was_scaled=True

    segments=[]
    cursor=0.0
    for f in features:
        gap=max(0.0,f['gap_before'])*scale
        width=max(0.0,f['width'])*scale
        if gap>1e-9:
            segments.append((ENV_MATERIAL,gap))
        if width>1e-9:
            segments.append((f['material'],width))
        cursor+=gap+width

    trailing=1.0-cursor
    if trailing>1e-9:
        segments.append((ENV_MATERIAL,trailing))

    return segments, was_scaled


def resolve_manual_block(block):
    segments,was_scaled=build_manual_segments(block['features'])
    return [{'thickness':block['thickness'],'segments':segments}], was_scaled


def resolve_pair_block(block):
    layers=[]
    any_scaled=False
    for _ in range(int(block['n_pairs'])):
        for side in ('a','b'):
            material=block[f'mat_{side}']
            width=block[f'w_{side}']
            thickness=block[f'd_{side}']
            gap=(1.0-width)/2.0
            segments,scaled=build_manual_segments([{'material':material,'width':width,'gap_before':gap}])
            any_scaled=any_scaled or scaled
            layers.append({'thickness':thickness,'segments':segments})
    return layers, any_scaled


def resolve_taper_block(block):
    n=max(1,int(block['n_slices']))
    layers=[]
    any_scaled=False
    for i in range(n):
        t=i/(n-1) if n>1 else 0.0
        width=block['top_width']+(block['bottom_width']-block['top_width'])*t
        width=max(0.0,min(1.0,width))
        if block['alignment']=='Left':
            gap=0.0
        elif block['alignment']=='Right':
            gap=1.0-width
        else:
            gap=(1.0-width)/2.0
        segments,scaled=build_manual_segments([{'material':block['material'],'width':width,'gap_before':gap}])
        any_scaled=any_scaled or scaled
        layers.append({'thickness':block['thickness']/n,'segments':segments})
    return layers, any_scaled


def resolve_pedestal_block(block):
    """
    基板ペデスタル（基板上げ）ブロック。
    基板と同じ材質 (SUBST_MATERIAL) を指定した幅・位置に配置した1層を生成する。
    これにより「基板が height nm 盛り上がった」台座構造を表現できる。
    """
    height = block['height']
    width  = max(0.0, min(1.0, block['fraction']))
    align  = block.get('alignment', 'Center')
    if align == 'Left':
        gap = 0.0
    elif align == 'Right':
        gap = 1.0 - width
    else:
        gap = (1.0 - width) / 2.0

    segments = []
    if gap > 1e-9:
        segments.append((ENV_MATERIAL, gap))
    if width > 1e-9:
        segments.append((SUBST_MATERIAL, width))
    trailing = 1.0 - gap - width
    if trailing > 1e-9:
        segments.append((ENV_MATERIAL, trailing))

    return [{'thickness': height, 'segments': segments}], False


def resolve_all_blocks(blocks):
    layers=[]
    any_scaled=False
    for block in blocks:
        if block['type']=='manual':
            block_layers,scaled=resolve_manual_block(block)
        elif block['type']=='pair':
            block_layers,scaled=resolve_pair_block(block)
        elif block['type']=='taper':
            block_layers,scaled=resolve_taper_block(block)
        elif block['type']=='pedestal':
            block_layers,scaled=resolve_pedestal_block(block)
        else:
            continue
        layers.extend(block_layers)
        any_scaled=any_scaled or scaled
    return layers, any_scaled


# 可視化用の関数（材料名で色分け、層内の任意個数のセグメントに対応）
def create_structure_figure(layers, pitch_nm, nk_subst_name):
    """設定された多層膜構造（ブロックから解決された層リスト）を可視化するPlotly Figureを生成する"""

    shapes = []
    annotations = []

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    unique_materials = sorted({mat for L in layers for mat, w in L['segments']
                               if mat not in (ENV_MATERIAL, SUBST_MATERIAL)})
    color_map = {material: colors[i % len(colors)] for i, material in enumerate(unique_materials)}
    env_color   = 'rgba(240, 240, 240, 0.95)'
    subst_color = 'rgba(160, 200, 230, 0.9)'  # 基板ペデスタル用（薄青）

    total_thickness = sum(L['thickness'] for L in layers)

    # 1. 基板を描画
    substrate_thickness = total_thickness * 0.2 if total_thickness > 0 else 50
    shapes.append(go.layout.Shape(
        type="rect", xref="x", yref="y",
        x0=0, y0=-substrate_thickness, x1=pitch_nm, y1=0,
        fillcolor="lightgrey", line_width=0
    ))
    annotations.append(dict(
        x=pitch_nm / 2, y=-substrate_thickness / 2,
        text=nk_subst_name, showarrow=False, font=dict(color="black")
    ))

    current_y = 0
    # 2. 各層を基板の上から順番に描画（layersはTopが先頭なので、逆順にして基板側から描く）
    for L in reversed(layers):
        d = L['thickness']
        cursor = 0.0
        for mat, w in L['segments']:
            x0 = pitch_nm * cursor
            x1 = pitch_nm * (cursor + w)
            if mat == ENV_MATERIAL:
                color = env_color
            elif mat == SUBST_MATERIAL:
                color = subst_color
            else:
                color = color_map.get(mat, '#cccccc')
            shapes.append(go.layout.Shape(
                type="rect", xref="x", yref="y",
                x0=x0, y0=current_y, x1=x1, y1=current_y + d,
                fillcolor=color, line=dict(color="black", width=1)
            ))
            if mat not in (ENV_MATERIAL, SUBST_MATERIAL):
                annotations.append(dict(
                    x=(x0 + x1) / 2, y=current_y + d / 2,
                    text=f"{mat}<br>d={d:g}nm", showarrow=False, font=dict(color="white", size=10)
                ))
            elif mat == SUBST_MATERIAL:
                annotations.append(dict(
                    x=(x0 + x1) / 2, y=current_y + d / 2,
                    text=f"基板<br>d={d:g}nm", showarrow=False, font=dict(color="black", size=9)
                ))
            cursor += w

        current_y += d

    fig = go.Figure()

    fig.update_layout(
        title="Film Stack Visualization",
        xaxis_title="Position (nm)",
        yaxis_title="Height (nm)",
        xaxis=dict(range=[0, pitch_nm], showgrid=False, zeroline=False),
        yaxis=dict(range=[-substrate_thickness, current_y * 1.1 if current_y > 0 else 100], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False),
        shapes=shapes,
        annotations=annotations,
        plot_bgcolor='white'
    )

    return fig


def get_layer_tuple(wl, layers, nk_fn_map, n_env_val, nk_subst_fn):
    """
    指定波長wl[nm]での layerを返す（layersは任意個数のセグメントを持つ層のリスト、Topが先頭）
    """
    n_env = complex(n_env_val)
    layer_env = (0, n_env, 0)                                   # 媒質層
    layer_subst = (0, complex(nk_subst_fn(wl)), 0)               # 基板層

    layer_list = [layer_subst]
    for L in reversed(layers):
        flat = []
        for mat, w in L['segments']:
            if w <= 1e-9:
                continue
            if mat == ENV_MATERIAL:
                n = n_env
            elif mat == SUBST_MATERIAL:
                n = complex(nk_subst_fn(wl))
            else:
                n = complex(nk_fn_map[mat](wl))
            flat.append(n)
            flat.append(w)
        if not flat:
            continue
        layer_list.append(tuple([L['thickness'] / 1000.0] + flat))

    layer_list.append(layer_env)
    return tuple(layer_list)


@tictoc
def calc_rcwa1d(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env_val, nk_subst_fn):
    nwl = len(wl_nm_ar)
    irp = np.empty([nwl, norder],dtype=float) # 反射回折効率(p)の格納用
    itp = np.empty([nwl, norder],dtype=float) # 透過回折効率(p)の格納用
    irs = np.empty([nwl, norder],dtype=float) # 反射回折効率(s)の格納用
    its = np.empty([nwl, norder],dtype=float) # 透過回折効率(s)の格納用

    for idx,wl_nm in enumerate(wl_nm_ar):
        wl_um=float(wl_nm/1000.0)
        layer=get_layer_tuple(float(wl_nm), layers, nk_fn_map, n_env_val, nk_subst_fn)
        coef=float(2*np.pi*np.sin(inc_angle_rad)/wl_um)
        irp[idx,:], itp[idx,:] = Rcwa1d_cached('p', wl_um, coef, pitch_um, layer, norder)    # RCWAの呼び出し
        irs[idx,:], its[idx,:] = Rcwa1d_cached('s', wl_um, coef, pitch_um, layer, norder)    # RCWAの呼び出し
    return (irp,itp,irs,its)




st.title('Structural color simulator (v2: 複合構造ビルダー)')

st.sidebar.header('Light parameters')

nk_namelist=get_nk_list()
if len(nk_namelist)<1:
    st.error('nk list not find')

nk_idx_subst=nk_namelist.index('Air')
nk_idx_subst=nk_namelist.index('Silicon')
nk_idx_film=nk_namelist.index('SiO2')


angle_mode=st.sidebar.radio('Angle mode',('Single angle (2D)','Angle sweep (3D)'),key='angle_mode')
if angle_mode=='Single angle (2D)':
    inc_angle=st.sidebar.number_input('Incident angle [deg]',min_value=0.0,max_value=89.0,value=0.0,step=0.1,format='%3.1f')
else:
    angle_range=st.sidebar.slider('Angle range [deg]',min_value=0.0,max_value=89.0,value=(0.0,80.0),step=1.0,format='%.0f')
    angle_step=st.sidebar.number_input('Angle step [deg]',min_value=0.5,max_value=45.0,value=2.0,step=0.5,format='%.1f')
    inc_angle=angle_range[0]   # 色計算・CSV用の代表角度（スイープ先頭）
spMenu=('Visible[380-780nm]','UV[200-400nm]','NIR[700-1000nm]','All[200-1000nm]','Any')
wl_option=st.sidebar.selectbox('Spetrum range',spMenu)
if wl_option==spMenu[0]:
    wl_min=380.0
    wl_max=780.0
    wl_n=81
elif wl_option==spMenu[1]:
    wl_min=200.0
    wl_max=400.0
    wl_n=101
elif wl_option==spMenu[2]:
    wl_min=700.0
    wl_max=1000.0
    wl_n=61
elif wl_option==spMenu[3]:
    wl_min=200.0
    wl_max=1000.0
    wl_n=81

if wl_option==spMenu[4]:
    wl_range=st.sidebar.slider('Wavelength range [nm]',min_value=200.0,max_value=1000.0,value=(wl_min,wl_max),step=20.0,format='%.0f')
    if wl_range:
        wl_min=wl_range[0]
        wl_max=wl_range[1]

    wl_n=st.sidebar.number_input('Number of Wavelength',min_value=11,max_value=101,value=int(wl_n),step=10,format='%d',key='wln')


st.sidebar.header('Atmosphere')
n_env=st.sidebar.number_input('Refractive index (air:1.00)',min_value=1.0,max_value=3.0,value=n_env,step=0.01,format='%3.2f',key='nenv')

st.sidebar.header('Substrate')
nk_subst_name=st.sidebar.selectbox('Substrate',nk_namelist,index=nk_idx_subst,key='substrate')




st.header('Structural color of patterned film using 1D RCWA/FMM')

st.subheader('Film parameters (Number of layers, Period(pitch), Order)')

col1,col2=st.columns(2)
with col1:
    pitch_nm=st.number_input('Pitch[nm]',min_value=1.0,max_value=1e6,value=pitch_nm,step=1.0,format='%g',key='Pitch')
with col2:
    norder=st.number_input('nOrder',min_value=5,max_value=101,value=norder,step=2,format='%d',key='nOrder')


# --- 複合構造ビルダー ---
# 「ブロック」を任意の順序・個数で追加して積み重ねる。先に追加したブロックほどTop（光が最初に当たる側）。
# - Manual layer: 1層分。セグメント(材質・幅・直前からのギャップ)を複数指定できるので多リッジ構造を表現できる。
#   幅の異なるManual layerを複数積めば段差(非対称な幅変化)も表現できる。
# - Periodic pair: 既存のA/B交互積層機能。
# - Tapered block: top/bottom幅を線形補間したN枚の薄層に自動分割し、台形(斜め側壁)を近似する。

if 'blocks' not in st.session_state:
    st.session_state.blocks = []
    st.session_state.next_block_id = 0

def _new_block_id():
    st.session_state.next_block_id += 1
    return st.session_state.next_block_id

st.subheader('Patterned film stack (blocks)')
st.caption('ブロックを追加して任意の順序で積み重ねられます。先に追加したブロックほど光が最初に当たる側（Top）になります。')

bcol1, bcol2, bcol3, bcol4 = st.columns(4)
with bcol1:
    if st.button('+ Manual layer (多リッジ・段差用)'):
        st.session_state.blocks.append({
            'id': _new_block_id(), 'type': 'manual',
            'thickness': 100.0,
            'features': [{'material': nk_namelist[nk_idx_film], 'width': 0.5, 'gap_before': 0.25}],
        })
with bcol2:
    if st.button('+ Periodic pair (A+B) x N'):
        idx_b = 1 if len(nk_namelist) > 1 else 0
        st.session_state.blocks.append({
            'id': _new_block_id(), 'type': 'pair',
            'n_pairs': 5,
            'mat_a': nk_namelist[nk_idx_film], 'w_a': 0.5, 'd_a': 100.0,
            'mat_b': nk_namelist[idx_b], 'w_b': 0.5, 'd_b': 100.0,
        })
with bcol3:
    if st.button('+ Tapered block (台形用)'):
        st.session_state.blocks.append({
            'id': _new_block_id(), 'type': 'taper',
            'material': nk_namelist[nk_idx_film],
            'top_width': 0.3, 'bottom_width': 0.7,
            'thickness': 200.0, 'n_slices': 10, 'alignment': 'Center',
        })
with bcol4:
    if st.button('+ Substrate pedestal (基板上げ)'):
        st.session_state.blocks.append({
            'id': _new_block_id(), 'type': 'pedestal',
            'height': 100.0, 'fraction': 0.4, 'alignment': 'Center',
        })

remove_idx = None
move = None

for i, block in enumerate(st.session_state.blocks):
    bid = block['id']
    with st.expander(f"Block {i + 1} [{block['type']}]", expanded=True):
        if block['type'] == 'manual':
            block['thickness'] = st.number_input('Thickness[nm]', min_value=0.0, max_value=1e6, value=block['thickness'], step=0.1, format='%g', key=f'th_{bid}')
            n_feat = st.number_input('Number of ridges/segments', min_value=1, max_value=6, value=len(block['features']), step=1, format='%d', key=f'nf_{bid}')
            while len(block['features']) < n_feat:
                block['features'].append({'material': nk_namelist[nk_idx_film], 'width': 0.2, 'gap_before': 0.1})
            while len(block['features']) > n_feat:
                block['features'].pop()

            for fi, feat in enumerate(block['features']):
                c1, c2, c3 = st.columns(3)
                with c1:
                    # 先頭に「基板材質」選択肢を追加（SUBST_MATERIAL センチネルにマップ）
                    _SUBST_LABEL = f'[基板材質: {nk_subst_name}]'
                    _mat_options = [_SUBST_LABEL] + nk_namelist
                    if feat['material'] == SUBST_MATERIAL:
                        _mat_idx = 0
                    elif feat['material'] in nk_namelist:
                        _mat_idx = nk_namelist.index(feat['material']) + 1
                    else:
                        _mat_idx = nk_idx_film + 1
                    _selected = st.selectbox(f'Material #{fi + 1}', _mat_options, index=_mat_idx, key=f'mat_{bid}_{fi}')
                    feat['material'] = SUBST_MATERIAL if _selected == _SUBST_LABEL else _selected
                with c2:
                    feat['width'] = st.number_input(f'Width #{fi + 1} (fraction of pitch)', min_value=0.0, max_value=1.0, value=feat['width'], step=0.001, format='%.3f', key=f'w_{bid}_{fi}')
                with c3:
                    feat['gap_before'] = st.number_input(f'Gap before #{fi + 1} (fraction of pitch)', min_value=0.0, max_value=1.0, value=feat['gap_before'], step=0.001, format='%.3f', key=f'gap_{bid}_{fi}')

        elif block['type'] == 'pair':
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('##### Layer A (top side of each pair)')
                block['mat_a'] = st.selectbox('Material A', nk_namelist, index=nk_namelist.index(block['mat_a']), key=f'mata_{bid}')
                block['w_a'] = st.number_input('Ratio A', 0.0, 1.0, block['w_a'], 0.001, key=f'wa_{bid}')
                block['d_a'] = st.number_input('Thickness A[nm]', 0.0, 1e6, block['d_a'], 0.1, key=f'da_{bid}')
            with c2:
                st.markdown('##### Layer B (bottom side of each pair)')
                block['mat_b'] = st.selectbox('Material B', nk_namelist, index=nk_namelist.index(block['mat_b']), key=f'matb_{bid}')
                block['w_b'] = st.number_input('Ratio B', 0.0, 1.0, block['w_b'], 0.001, key=f'wb_{bid}')
                block['d_b'] = st.number_input('Thickness B[nm]', 0.0, 1e6, block['d_b'], 0.1, key=f'db_{bid}')
            block['n_pairs'] = st.number_input('Number of pairs', min_value=1, max_value=50, value=block['n_pairs'], step=1, format='%d', key=f'np_{bid}')

        elif block['type'] == 'taper':
            block['material'] = st.selectbox('Material', nk_namelist, index=nk_namelist.index(block['material']), key=f'tmat_{bid}')
            c1, c2, c3 = st.columns(3)
            with c1:
                block['top_width'] = st.number_input('Top width (fraction)', 0.0, 1.0, block['top_width'], 0.001, key=f'tw_{bid}')
            with c2:
                block['bottom_width'] = st.number_input('Bottom width (fraction)', 0.0, 1.0, block['bottom_width'], 0.001, key=f'bw_{bid}')
            with c3:
                block['n_slices'] = st.number_input('Number of slices', min_value=1, max_value=100, value=block['n_slices'], step=1, format='%d', key=f'ns_{bid}')
            block['thickness'] = st.number_input('Total thickness[nm]', min_value=0.0, max_value=1e6, value=block['thickness'], step=0.1, format='%g', key=f'tt_{bid}')
            block['alignment'] = st.radio('Alignment', ('Left', 'Center', 'Right'), index=('Left', 'Center', 'Right').index(block['alignment']), key=f'al_{bid}', horizontal=True)

        elif block['type'] == 'pedestal':
            st.info(f'基板と同じ材質（サイドバーで選択中: **{nk_subst_name}**）を配置します。')
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                block['height'] = st.number_input('Raised height[nm]', min_value=0.0, max_value=1e6,
                                                  value=block['height'], step=0.1, format='%g', key=f'ph_{bid}')
            with pc2:
                block['fraction'] = st.number_input('Width fraction (0–1)', min_value=0.0, max_value=1.0,
                                                    value=block['fraction'], step=0.001, format='%.3f', key=f'pf_{bid}')
            with pc3:
                block['alignment'] = st.radio('Alignment', ('Left', 'Center', 'Right'),
                                              index=('Left', 'Center', 'Right').index(block.get('alignment', 'Center')),
                                              key=f'pal_{bid}', horizontal=True)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            if st.button('▲ Move up', key=f'up_{bid}', disabled=(i == 0)):
                move = (i, -1)
        with mcol2:
            if st.button('▼ Move down', key=f'down_{bid}', disabled=(i == len(st.session_state.blocks) - 1)):
                move = (i, 1)
        with mcol3:
            if st.button('✕ Remove block', key=f'rm_{bid}'):
                remove_idx = i

if remove_idx is not None:
    st.session_state.blocks.pop(remove_idx)
    st.rerun()
if move is not None:
    i, d = move
    j = i + d
    st.session_state.blocks[i], st.session_state.blocks[j] = st.session_state.blocks[j], st.session_state.blocks[i]
    st.rerun()

layers, was_scaled = resolve_all_blocks(st.session_state.blocks)
if was_scaled:
    st.warning('一部のブロックでセグメント幅の合計が1.0(ピッチ)を超えていたため、自動的に縮小しました。')

nlayers = len(layers)
st.caption(f'Resolved into {nlayers} physical layer(s) for RCWA calculation.')


st.subheader('Film Stack Visualization')
fig_structure = create_structure_figure(layers, pitch_nm, nk_subst_name)
st.plotly_chart(fig_structure, use_container_width=True)


st.subheader('Spectrum')

material_names = [mat for L in layers for mat, w in L['segments'] if mat != ENV_MATERIAL]
nk_fn_map = build_nk_fn_map(material_names)
nk_subst_fn = make_nk_fn([nk_subst_name])[0]

pitch_um=pitch_nm/1000.0
wl_nm_ar=np.linspace(wl_min,wl_max,wl_n,dtype=float)
idex_0=norder // 2
nmax_order=idex_0
order_list=list(range(-nmax_order, nmax_order+1))

def _order_label(m):
    return '0 (specular)' if m==0 else f'{m:+d}'

def _idx_for_order(m):
    return idex_0 + m

st.markdown('##### Diffraction order')
ocol1, ocol2 = st.columns(2)
with ocol1:
    plot_orders=st.multiselect('Orders to show in spectrum plot', order_list, default=[0], format_func=_order_label, key='plot_orders')
    if not plot_orders:
        plot_orders=[0]
with ocol2:
    repr_order=st.selectbox('Order for colorimetry / CSV / 3D surface', order_list, index=order_list.index(0), format_func=_order_label, key='repr_order')

if angle_mode=='Single angle (2D)':
    inc_angle_rad=inc_angle*np.pi/180.0
    (irp, itp,irs,its)=calc_rcwa1d(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn)

    Rp=irp[:,_idx_for_order(repr_order)]
    Rs=irs[:,_idx_for_order(repr_order)]
    Tp=itp[:,_idx_for_order(repr_order)]
    Ts=its[:,_idx_for_order(repr_order)]

    # 0次光以外の全次数の合算スペクトル（回折光の色計算用）
    _nonzero_orders=[m for m in order_list if m != 0]
    if _nonzero_orders:
        Rp_diff=np.clip(sum(irp[:,_idx_for_order(m)] for m in _nonzero_orders), 0.0, 1.0)
        Rs_diff=np.clip(sum(irs[:,_idx_for_order(m)] for m in _nonzero_orders), 0.0, 1.0)
    else:
        Rp_diff=np.zeros(len(wl_nm_ar))
        Rs_diff=np.zeros(len(wl_nm_ar))

    fig = go.Figure()

    for m in sorted(plot_orders):
        idx_m=_idx_for_order(m)
        fig.add_trace(go.Scatter(
            x=wl_nm_ar, y=irp[:,idx_m],
            name=f'Rp (m={_order_label(m)})',
            mode='lines',
        ))
        fig.add_trace(go.Scatter(
            x=wl_nm_ar, y=irs[:,idx_m],
            name=f'Rs (m={_order_label(m)})',
            mode='lines',
        ))

    title_msg=f'AOI at {round(inc_angle,1)}[deg]'
    fig.update_layout(title=title_msg,
                    yaxis_zeroline=True, xaxis_zeroline=True)
    fig.update_xaxes(title_text='Wavelength(nm)')

    # 0次光のみの場合は物理的上限[0,1]を維持、高次光を含む場合は自動スケール（データに合わせて拡大）
    if plot_orders == [0]:
        fig.update_yaxes(title_text='R/T', range=[0, 1])
    else:
        all_y = np.concatenate([irp[:, _idx_for_order(m)] for m in plot_orders] +
                               [irs[:, _idx_for_order(m)] for m in plot_orders])
        y_top = max(float(all_y.max()) * 1.15, 1e-3)
        fig.update_yaxes(title_text='R/T', range=[0, min(y_top, 1.0)])
        if y_top < 0.1:
            st.info(f'高次回折光（m≠0）の反射率は通常0次光より大幅に小さい値になります。現在の最大値: {all_y.max():.4f}')

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f'※ 下の色計算・CSVは回折次数 m={_order_label(repr_order)} のスペクトルに基づきます。')

else:
    # 角度配列を作成（範囲＋刻み）
    angle_ar=np.arange(angle_range[0], angle_range[1]+angle_step/2.0, angle_step, dtype=float)
    nang=len(angle_ar)

    Rp2d=np.empty([nang, wl_n],dtype=float)
    Rs2d=np.empty([nang, wl_n],dtype=float)
    Tp2d=np.empty([nang, wl_n],dtype=float)
    Ts2d=np.empty([nang, wl_n],dtype=float)
    Rp_diff_2d=np.zeros([nang, wl_n],dtype=float)
    Rs_diff_2d=np.zeros([nang, wl_n],dtype=float)

    idx_repr=_idx_for_order(repr_order)
    _nonzero_orders=[m for m in order_list if m != 0]
    _nonzero_idx=[_idx_for_order(m) for m in _nonzero_orders]
    prog=st.progress(0.0, text='Calculating angle sweep...')
    for ia, ang in enumerate(angle_ar):
        ang_rad=ang*np.pi/180.0
        (irp, itp, irs, its)=calc_rcwa1d(wl_nm_ar, ang_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn)
        Rp2d[ia,:]=irp[:,idx_repr]
        Rs2d[ia,:]=irs[:,idx_repr]
        Tp2d[ia,:]=itp[:,idx_repr]
        Ts2d[ia,:]=its[:,idx_repr]
        if _nonzero_idx:
            Rp_diff_2d[ia,:]=np.clip(irp[:,_nonzero_idx].sum(axis=1), 0.0, 1.0)
            Rs_diff_2d[ia,:]=np.clip(irs[:,_nonzero_idx].sum(axis=1), 0.0, 1.0)
        prog.progress((ia+1)/nang, text=f'Calculating angle sweep... {ia+1}/{nang}')
    prog.empty()

    pol_sel=st.radio('Polarization (3D surface)',('Rp','Rs'),horizontal=True,key='pol3d')
    Z = Rp2d if pol_sel=='Rp' else Rs2d
    fig = go.Figure(data=[go.Surface(
        x=wl_nm_ar, y=angle_ar, z=Z,
        colorscale='Viridis', cmin=0.0, cmax=1.0,
        colorbar=dict(title='R')
    )])
    fig.update_layout(
        title=f'{pol_sel} (m={_order_label(repr_order)}): Reflectance vs Wavelength & Angle',
        scene=dict(
            xaxis_title='Wavelength(nm)',
            yaxis_title='Angle(deg)',
            zaxis=dict(title='R', range=[0,1]),
        ),
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 角度依存グラフ（波長ごとの反射率 vs 角度）---
    st.subheader('Reflectance vs Angle (per wavelength)')

    # 表示波長の選択肢：wl_nm_ar から整数丸めで重複除去
    wl_choices = sorted(list(dict.fromkeys([int(round(w)) for w in wl_nm_ar])))

    # デフォルト波長と対応色
    _PRESET_WLS    = [400, 470, 540, 600, 700]
    _PRESET_COLORS = ['#8B00FF', '#0000FF', '#008000', '#FFD700', '#FF0000']
    _WL_COLOR_MAP  = dict(zip(_PRESET_WLS, _PRESET_COLORS))

    import numpy as _np
    default_wls = [w for w in _PRESET_WLS if w in wl_choices]

    acol1, acol2 = st.columns(2)
    with acol1:
        sel_wls = st.multiselect(
            'Wavelengths to plot [nm]', wl_choices,
            default=default_wls, key='sel_wls_angle'
        )
    with acol2:
        pol_angle = st.radio(
            'Polarization', ('Rp', 'Rs'),
            horizontal=True, key='pol_angle'
        )

    if sel_wls:
        Z_ang = Rp2d if pol_angle == 'Rp' else Rs2d

        # 角度軸を対称に折り返す（計算なし、データのミラー）
        # angle_ar が 0 始まりの場合は 0 を重複させない
        if float(angle_ar[0]) == 0.0:
            _x_sym = _np.concatenate([-angle_ar[1:][::-1], angle_ar])
            def _mirror(col): return _np.concatenate([col[1:][::-1], col])
        else:
            _x_sym = _np.concatenate([-angle_ar[::-1], angle_ar])
            def _mirror(col): return _np.concatenate([col[::-1], col])

        fig_ang = go.Figure()
        _auto_colors = ['#17becf','#e377c2','#7f7f7f','#bcbd22','#8c564b']
        _auto_idx = 0
        for wl_sel in sorted(sel_wls):
            idx_wl = int(_np.argmin(_np.abs(wl_nm_ar - wl_sel)))
            actual_wl = int(round(wl_nm_ar[idx_wl]))
            color = _WL_COLOR_MAP.get(wl_sel)
            if color is None:
                color = _auto_colors[_auto_idx % len(_auto_colors)]
                _auto_idx += 1
            fig_ang.add_trace(go.Scatter(
                x=_x_sym, y=_mirror(Z_ang[:, idx_wl]),
                name=f'{actual_wl} nm',
                mode='lines',
                line=dict(color=color, width=2),
            ))
        fig_ang.update_layout(
            title=f'{pol_angle} (m={_order_label(repr_order)}): Reflectance vs Angle',
            xaxis_title='Angle (deg)',
            yaxis_title='Reflectance',
        )
        fig_ang.update_xaxes(range=[float(-angle_ar[-1]), float(angle_ar[-1])])
        fig_ang.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_ang, use_container_width=True)

    # 下流の色計算・CSV用に代表角度（先頭）のスペクトルを1Dとして用意
    inc_angle=float(angle_ar[0])
    Rp=Rp2d[0,:]; Rs=Rs2d[0,:]; Tp=Tp2d[0,:]; Ts=Ts2d[0,:]
    Rp_diff=Rp_diff_2d[0,:]; Rs_diff=Rs_diff_2d[0,:]
    st.caption(f'※ 下の色計算・CSVは代表角度 {round(inc_angle,1)}°・回折次数 m={_order_label(repr_order)} のスペクトルに基づきます。')

if wl_option==spMenu[0]:
    st.subheader('Colorimetry')
    if repr_order != 0:
        st.warning(f'色計算は回折次数 m={_order_label(repr_order)} のスペクトルを使用しています。高次回折光は強度が非常に小さいため、色は暗く（黒に近く）なります。0次光（specular）の色を見たい場合は "Order for colorimetry" を 0 に戻してください。')

    colour_Rp,colour_Rs,colour_Tp,colour_Ts,colour_wl_ar=Rp,Rs,Tp,Ts,wl_nm_ar

    def _make_sd(arr, name):
        sd = colour.SpectralDistribution(arr, name=name)
        sd.wavelengths = wl_nm_ar
        return sd

    def _sd_to_hex(arr, name):
        sd = _make_sd(arr, name)
        cmfs_ = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        ill_  = colour.SDS_ILLUMINANTS['D65']
        XYZ   = colour.sd_to_XYZ(sd, cmfs_, ill_)
        RGB   = colour.XYZ_to_sRGB(XYZ / 100)
        b     = [int(np.clip(round(v*255), 0, 255)) for v in RGB]
        hex_  = '#'+format(b[0],'02x')+format(b[1],'02x')+format(b[2],'02x')
        return XYZ, hex_

    # Convert to Tristimulus Values
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS['D65']

    XYZ_Rp, strRGB_Rp = _sd_to_hex(colour_Rp, 'Rp')
    XYZ_Rs, strRGB_Rs = _sd_to_hex(colour_Rs, 'Rs')
    XYZ_Rp_diff, strRGB_Rp_diff = _sd_to_hex(Rp_diff, 'Rp_diff')
    XYZ_Rs_diff, strRGB_Rs_diff = _sd_to_hex(Rs_diff, 'Rs_diff')

    st.caption(f'「回折光の色」は0次光以外の全次数（m≠0）の反射率を合算したスペクトルから算出しています。ピッチが波長より小さい場合は高次光が伝搬できないため黒になります（ピッチ≧波長 が必要）。')

    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.color_picker(f'Rp (m={_order_label(repr_order)})', strRGB_Rp, key='cp_Rp'+strRGB_Rp)
        st.write('XYZ', XYZ_Rp)
    with col2:
        st.color_picker(f'Rs (m={_order_label(repr_order)})', strRGB_Rs, key='cp_Rs'+strRGB_Rs)
        st.write('XYZ', XYZ_Rs)
    with col3:
        st.color_picker('Rp 回折光 (m≠0 合算)', strRGB_Rp_diff, key='cp_Rp_diff'+strRGB_Rp_diff)
        st.write('XYZ', XYZ_Rp_diff)
    with col4:
        st.color_picker('Rs 回折光 (m≠0 合算)', strRGB_Rs_diff, key='cp_Rs_diff'+strRGB_Rs_diff)
        st.write('XYZ', XYZ_Rs_diff)




t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
d = now.strftime('%Y%m%d%H%M%S')

st.subheader('Download spectrum and color data')

# --- 反射率スペクトル CSV（透過光なし）---
nwl=len(wl_nm_ar)
df_spec=pd.DataFrame(
    np.stack([Rp, Rs], axis=1),
    index=wl_nm_ar,
    columns=['Rp', 'Rs']
)
df_spec.index.name='Wavelength(nm)'
csv_spec = convert_df(df_spec)
st.download_button(
    label='Download reflectance spectrum CSV (Rp, Rs)',
    data=csv_spec,
    file_name=f'spectrum_{d}.csv',
    mime='text/csv',
)

# --- 角度依存反射率 CSV（角度スイープモードのみ）---
if angle_mode=='Angle sweep (3D)':
    # 対称角度軸を構築（グラフと同じミラー処理）
    if float(angle_ar[0]) == 0.0:
        _x_csv = np.concatenate([-angle_ar[1:][::-1], angle_ar])
        def _mir(col): return np.concatenate([col[1:][::-1], col])
    else:
        _x_csv = np.concatenate([-angle_ar[::-1], angle_ar])
        def _mir(col): return np.concatenate([col[::-1], col])

    wl_cols = [f'{int(round(w))}nm' for w in wl_nm_ar]

    df_ang_p = pd.DataFrame(
        {c: _mir(Rp2d[:, i]) for i, c in enumerate(wl_cols)},
        index=_x_csv
    )
    df_ang_p.index.name = 'Angle(deg)'

    df_ang_s = pd.DataFrame(
        {c: _mir(Rs2d[:, i]) for i, c in enumerate(wl_cols)},
        index=_x_csv
    )
    df_ang_s.index.name = 'Angle(deg)'

    csv_ang_p = convert_df(df_ang_p)
    csv_ang_s = convert_df(df_ang_s)

    acsvCol1, acsvCol2 = st.columns(2)
    with acsvCol1:
        st.download_button(
            label='Download angle sweep CSV (Rp, m={})'.format(_order_label(repr_order)),
            data=csv_ang_p,
            file_name=f'angle_Rp_{d}.csv',
            mime='text/csv',
        )
    with acsvCol2:
        st.download_button(
            label='Download angle sweep CSV (Rs, m={})'.format(_order_label(repr_order)),
            data=csv_ang_s,
            file_name=f'angle_Rs_{d}.csv',
            mime='text/csv',
        )
    st.caption('角度CSVの行 = 角度 (−{}° ～ +{}°)、列 = 各波長の反射率'.format(
        int(angle_ar[-1]), int(angle_ar[-1])))

print(Rp)

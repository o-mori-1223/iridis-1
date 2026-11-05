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
import rcwa_mh 
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


# 可視化用の新しい関数（色分けロジックを改善）
def create_structure_figure(d_list, w_list, nk_name_list, pitch_nm, nk_subst_name):
    """設定された多層膜構造を可視化するPlotly Figureを生成する（材料名で色分け）"""
    
    shapes = []
    annotations = []
    
    # 色のリスト
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # --- ここからが改善部分 ---
    # 材料名と色を対応させる辞書（カラーマップ）を作成
    unique_materials = sorted(list(set(nk_name_list)))
    color_map = {material: colors[i % len(colors)] for i, material in enumerate(unique_materials)}
    # --- ここまで ---

    # 1. 基板を描画
    substrate_thickness = sum(d_list) * 0.2 if d_list else 50 # d_listが空の場合のデフォルト値
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
    # 2. 各層を基板の上から順番に描画（入力はTopからなので、逆順にして基板側から描く）
    # reversed() を使ってリストを後ろから処理します
    for i, (d, w, name) in enumerate(zip(reversed(d_list), reversed(w_list), reversed(nk_name_list))):
        
        # --- ここも改善部分 ---
        # カラーマップから材料名に対応する色を取得
        color = color_map.get(name, '#cccccc') # もしマップになければ灰色
        # --- ここまで ---
        
        # 膜（ライン部分）
        shapes.append(go.layout.Shape(
            type="rect", xref="x", yref="y",
            x0=0, y0=current_y, x1=pitch_nm * w, y1=current_y + d,
            fillcolor=color, line=dict(color="black", width=1)
        ))
        # 媒質（ライン以外の部分）
        shapes.append(go.layout.Shape(
            type="rect", xref="x", yref="y",
            x0=pitch_nm * w, y0=current_y, x1=pitch_nm, y1=current_y + d,
            fillcolor="rgba(240, 240, 240, 0.95)", line=dict(color="black", width=1)
        ))
        
        # テキストラベルを追加
        annotations.append(dict(
            x=pitch_nm * w / 2, y=current_y + d / 2,
            text=f"{name}<br>d={d}nm", showarrow=False, font=dict(color="white", size=10)
        ))
        
        current_y += d
        
    fig = go.Figure()
    
    # レイアウト設定
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

#@functools.cache
def get_layer_tuple(wl,nk_fn_list,w_list,d_list,nkes_fn_list):
    """
    指定波長wl[nm]での layerを返す
    """
    layer_list=[]
    n_env=nkes_fn_list[0](wl)
    layer_env=(0,n_env,0)                               # 媒質層
    layer_subst=(0,complex(nkes_fn_list[1](wl)),0)      # 基板層
    layer_list.append(layer_subst)
    n=len(d_list)
    for k in range(n-1,-1,-1):
        nk=complex(nk_fn_list[k](wl))
        w=w_list[k]
        # layer=(d_list[k]/1000.0, nk,w, n_env, 1-w)
        layer=(d_list[k]/1000.0, nk,w/2.0, n_env, 1-w, nk, w/2.0)
        layer_list.append(layer)
    
    layer_list.append(layer_env)
    #print(layer_list)
    return tuple(layer_list)

@tictoc
def calc_rcwa1d(wl_nm_ar, inc_angle_rad, pitch_um, norder,nk_name_list,w_list,d_list,nkes_name_list):
    nkes_fn_list=make_nk_fn(nkes_name_list)
    nk_fn_list=make_nk_fn(nk_name_list)

    nwl = len(wl_nm_ar)
    irp = np.empty([nwl, norder],dtype=float) # 反射回折効率(p)の格納用
    itp = np.empty([nwl, norder],dtype=float) # 透過回折効率(p)の格納用
    irs = np.empty([nwl, norder],dtype=float) # 反射回折効率(s)の格納用
    its = np.empty([nwl, norder],dtype=float) # 透過回折効率(s)の格納用

    for idx,wl_nm in enumerate(wl_nm_ar):
        wl_um=float(wl_nm/1000.0)
        layer=get_layer_tuple(float(wl_nm),nk_fn_list,w_list,d_list,nkes_fn_list)
        coef=float(2*np.pi*np.sin(inc_angle_rad)/wl_um)
        irp[idx,:], itp[idx,:] = Rcwa1d_cached('p', wl_um, coef, pitch_um, layer, norder)    # RCWAの呼び出し
        irs[idx,:], its[idx,:] = Rcwa1d_cached('s', wl_um, coef, pitch_um, layer, norder)    # RCWAの呼び出し
    return (irp,itp,irs,its)







st.title('Structural color simulator')

st.sidebar.header('Light parameters')

nk_namelist=get_nk_list()
if len(nk_namelist)<1:
    st.error('nk list not find')

nk_idx_subst=nk_namelist.index('Air')
nk_idx_subst=nk_namelist.index('Silicon')
nk_idx_film=nk_namelist.index('SiO2')


inc_angle=st.sidebar.number_input('Incident angle [deg]',min_value=0.0,max_value=89.0,value=0.0,step=0.1,format='%3.1f')
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


nk_name_list=[]
w_list=[]
d_list=[]

# --- モードB: ペア繰り返し入力 (新機能) ---
    
    # 繰り返し回数
input_mode = st.radio("Layer Input Mode", ('Manual Input (Layer by Layer)', 'Periodic Pairs (Layer A + B) x N'))
if input_mode == 'Manual Input (Layer by Layer)':
    nlayers = st.number_input('Number of layer', min_value=1, max_value=100, value=nlayers, step=1, format='%d', key='nLayer')
    
    st.subheader('Patterned film stack')
    for num in range(nlayers):
        col1, col2, col3 = st.columns((1, 1, 1))
        # label_layer=order_n(num+1)+' layer' 
        label_layer = f'{num + 1} Layer (Top)'
        with col1:
            nk_name = st.selectbox(label_layer, nk_namelist, index=nk_idx_film, key='L'+str(num+1))
            nk_name_list.append(nk_name)
        with col2:
            val = st.number_input('Fill ratio', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format='%.3f', key='R'+str(num+1))
            w_list.append(val)
        with col3:
            val = st.number_input('thickness[nm]', min_value=0.0, max_value=1e6, value=100.0, step=0.1, format='%g', key='T'+str(num+1))
            d_list.append(val)

else:
    n_pairs = st.number_input('Number of Pairs', min_value=1, max_value=50, value=5, step=1, format='%d')
        
    st.markdown("---")
    col1, col2 = st.columns(2)
        
        # 層A (Top側)
    with col1:
            st.markdown("##### Layer A (top)")
            nk_a = st.selectbox("Material A", nk_namelist, index=nk_idx_film, key='pair_nk_a')
            w_a = st.number_input("Ratio A", 0.0, 1.0, 0.5, 0.001, key='pair_w_a')
            d_a = st.number_input("Thickness A [nm]", 0.0, 1e6, 100.0, 0.1, key='pair_d_a')
            
        # 層B (Bottom側)
    with col2:
            st.markdown("##### Layer B (bottom)")
            # デフォルトでリストの2番目などを選ばせる処理（エラー回避）
            idx_b = 0
            if len(nk_namelist) > 1: idx_b = 1
            
            nk_b = st.selectbox("Material B", nk_namelist, index=idx_b, key='pair_nk_b')
            w_b = st.number_input("Ratio B", 0.0, 1.0, 0.5, 0.001, key='pair_w_b')
            d_b = st.number_input("Thickness B [nm]", 0.0, 1e6, 100.0, 0.1, key='pair_d_b')

    # リストの増殖

    nk_name_list = [nk_a, nk_b] * n_pairs
    w_list = [w_a, w_b] * n_pairs
    d_list = [d_a, d_b] * n_pairs
        
    nlayers = len(nk_name_list)
    st.success(f"Generated {nlayers} layers ({n_pairs} pairs).")



# st.subheader('Spectrum')





# ... (forループの後) ...

# --- ここから追加 ---
st.subheader('Film Stack Visualization')
# 可視化関数を呼び出し
fig_structure = create_structure_figure(d_list, w_list, nk_name_list, pitch_nm, nk_subst_name)
# Streamlitでグラフを表示
st.plotly_chart(fig_structure, use_container_width=True)
# --- ここまで追加 ---


st.subheader('Spectrum')

# ... (以降のコードはそのまま) ...

nkes_name_list=[n_env,nk_subst_name]

inc_angle_rad=inc_angle*np.pi/180.0
pitch_um=pitch_nm/1000.0
wl_nm_ar=np.linspace(wl_min,wl_max,wl_n,dtype=float)

(irp, itp,irs,its)=calc_rcwa1d(wl_nm_ar, inc_angle_rad, pitch_um, norder,nk_name_list,w_list,d_list,nkes_name_list)


idex_0=norder // 2

Rp=irp[:,idex_0]
Rs=irs[:,idex_0]
Tp=itp[:,idex_0]
Ts=its[:,idex_0]

# Rp=np.sum(irp,axis=1)
# Rs=np.sum(irs,axis=1)
# Tp=np.sum(itp,axis=1)
# Ts=np.sum(its,axis=1)



fig = go.Figure()

gkind='Reflectance'

fig.add_trace(go.Scatter(
    x=wl_nm_ar, y=Rp,
    name='Rp',
    mode='lines',
    marker_color='rgba(255, 0, 0, .8)'
))
fig.add_trace(go.Scatter(
    x=wl_nm_ar, y=Rs,
    name='Rs',
    mode='lines',
    marker_color='rgba(0, 255, 0, .8)'
))
#fig.add_trace(go.Scatter(
    #x=wl_nm_ar, y=Tp,
    #name='Tp',
    #mode='lines',
    #marker_color='rgba(0, 255, 255, .8)'
#))
#fig.add_trace(go.Scatter(
    #x=wl_nm_ar, y=Ts,
    #name='Ts',
    #mode='lines',
    #marker_color='rgba(0, 0, 255, .8)'
#))

# Set options common to all traces with fig.update_traces
#fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
title_msg=f'AOI at {round(inc_angle,1)}[deg]'
fig.update_layout(title=title_msg,
                yaxis_zeroline=True, xaxis_zeroline=True)
#fig.update_layout(legend_title_text = "Contestant")
fig.update_xaxes(title_text='Wavelength(nm)')
fig.update_yaxes(title_text='R/T',range=[0, 1])

st.plotly_chart(fig, use_container_width=True)

if wl_option==spMenu[0]:
    st.subheader('Colorimetry')


    colour_Rp,colour_Rs,colour_Tp,colour_Ts,colour_wl_ar=Rp,Rs,Tp,Ts,wl_nm_ar

    sd_Rp = colour.SpectralDistribution(colour_Rp, name='Rp')    
    sd_Rs = colour.SpectralDistribution(colour_Rs, name='Rs')
    # sd_Tp = colour.SpectralDistribution(colour_Tp, name='Tp')    
    # sd_Ts = colour.SpectralDistribution(colour_Ts, name='Ts')
    
    sd_Rp.wavelengths=wl_nm_ar
    sd_Rs.wavelengths=wl_nm_ar
    # sd_Tp.wavelengths=wl_nm_ar
    # sd_Ts.wavelengths=wl_nm_ar
    
    
    # Convert to Tristimulus Values
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS['D65']

    # Calculating the sample spectral distribution *CIE XYZ* tristimulus values.
    XYZ_Rp = colour.sd_to_XYZ(sd_Rp, cmfs, illuminant)
    XYZ_Rs = colour.sd_to_XYZ(sd_Rs, cmfs, illuminant)
    # XYZ_Tp = colour.sd_to_XYZ(sd_Tp, cmfs, illuminant)
    # XYZ_Ts = colour.sd_to_XYZ(sd_Ts, cmfs, illuminant)
    RGB_Rp = colour.XYZ_to_sRGB(XYZ_Rp / 100)
    RGB_Rs = colour.XYZ_to_sRGB(XYZ_Rs / 100)
    # RGB_Tp = colour.XYZ_to_sRGB(XYZ_Tp / 100)
    # RGB_Ts = colour.XYZ_to_sRGB(XYZ_Ts / 100)
    b_Rp=[]
    for v in RGB_Rp:
        b_Rp.append(np.clip(round(v*255),0,255))
    b_Rs=[]
    for v in RGB_Rs:
        b_Rs.append(np.clip(round(v*255),0,255))
    # b_Tp=[]
    # for v in RGB_Tp:
    #     b_Tp.append(np.clip(round(v*255),0,255))
    # b_Ts=[]
    # for v in RGB_Ts:
    #     b_Ts.append(np.clip(round(v*255),0,255))


    strRGB_Rp='#'+format(b_Rp[0], '02x')+format(b_Rp[1], '02x')+format(b_Rp[2], '02x')
    strRGB_Rs='#'+format(b_Rs[0], '02x')+format(b_Rs[1], '02x')+format(b_Rs[2], '02x')
    # strRGB_Tp='#'+format(b_Tp[0], '02x')+format(b_Tp[1], '02x')+format(b_Tp[2], '02x')
    # strRGB_Ts='#'+format(b_Ts[0], '02x')+format(b_Ts[1], '02x')+format(b_Ts[2], '02x')

    col1,col2,col3,col4=st.columns(4)
    with col1:
        color_Rp = st.color_picker('Rp', strRGB_Rp,key='cp_Rp'+ strRGB_Rp)
        st.write('XYZ chromaticity',XYZ_Rp)
    with col2:
        color_Rs = st.color_picker('Rs', strRGB_Rs,key='cp_Rs'+ strRGB_Rs)
        st.write('XYZ chromaticity',XYZ_Rs)
    # with col3:
    #     color_Tp = st.color_picker('Tp', strRGB_Tp,key='cp_Tp')
    #     st.write('XYZ chromaticity',XYZ_Tp)
    # with col4:
    #     color_Ts = st.color_picker('Ts', strRGB_Ts,key='cp_Ts')
    #     st.write('XYZ chromaticity',XYZ_Ts)




nwl=len(wl_nm_ar)
data=np.concatenate([wl_nm_ar.reshape([nwl,1]),Rp.reshape([nwl,1]),Rs.reshape([nwl,1]),Tp.reshape([nwl,1]),Ts.reshape([nwl,1])],1)
df=pd.DataFrame(data,columns=['Wavelength(nm)', 'Rp', 'Rs', 'Tp', 'Ts'])
#df=df.reset_index(drop=True)
df=df.set_index('Wavelength(nm)')

csv = convert_df(df)

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
# YYYYMMDDhhmmss形式に書式化
d = now.strftime('%Y%m%d%H%M%S')
fname='data_'+d+'.csv'

st.subheader('Download spectrum and color data')

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=fname,
    mime='text/csv',
)

print(Rp)
#print(Rp0)
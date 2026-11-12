"""1D/2D両モード共通のヘルパー。"""
import os

import numpy as np
import streamlit as st
from scipy.interpolate import interp1d

from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL

NK_DIR = os.path.join('data', 'nk')

# 1D/2D 画面の本体のタブ（st.tabs の key。計算ボタンや「読み込んで再計算」で結果のタブへ切り替える）
TAB_KEYS = {'1d': 'c1d_tab', '2d': 'a2d_tab'}
TAB_STRUCT = '🧱 構造'
TAB_PREVIEW = '🔍 形状プレビュー'
TAB_CHECK = '⏱ 見積もり・収束確認'
TAB_RESULT = '📈 計算結果'


def main_tabs(mode, labels):
    """本体のタブ。選んだタブを session_state に持つので、show_result_tab で切り替えられ、
    .open で開いていないタブの重い描画（3D の図など）を省ける。"""
    return st.tabs(labels, key=TAB_KEYS[mode], on_change='rerun')


def show_result_tab(mode):
    """結果のタブを開く（ボタンの on_click や、タブを作る前のコールバックから呼ぶ）。"""
    st.session_state[TAB_KEYS[mode]] = TAB_RESULT


def compact_page_css():
    """画面上部の余白と見出しを小さくして、縦に入る量を増やす。"""
    st.html("""
    <style>
    [data-testid="stMainBlockContainer"] { padding-top: 4.25rem; padding-bottom: 6rem; }   /* ヘッダー（約 3.75rem）より少し大きく */
    [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
    [data-testid="stMainBlockContainer"] h1 { font-size: 1.6rem; padding: 0 0 0.3rem 0; }
    </style>
    """)


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


@st.cache_data
def get_nk_list():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(NK_DIR)
                  if os.path.isfile(os.path.join(NK_DIR, f)))


def make_nk_fn(nk_name_list=()):
    """材質名（data/nk/<名前>.nk）または数値 → `wl_nm -> complex` の関数のリスト。"""
    fns = []
    for name in nk_name_list:
        if isinstance(name, (complex, float, int)):
            fns.append(lambda wl, _n=complex(name): _n)
        elif isinstance(name, str) and name.isnumeric():
            fns.append(lambda wl, _n=complex(float(name)): _n)
        else:
            fpath = os.path.join(NK_DIR, name + '.nk')
            if os.path.isfile(fpath):
                d = np.loadtxt(fpath, comments=';', encoding='utf-8_sig')
                fns.append(interp1d(d[:, 0], d[:, 1] + d[:, 2]*1j, kind='linear', fill_value='extrapolate'))
            else:
                try:
                    nk = complex(name)
                except ValueError:
                    nk = complex(1.0)
                fns.append(lambda wl, _n=nk: _n)
    return fns


def build_nk_fn_map(material_names):
    uniq = list(dict.fromkeys(n for n in material_names if n not in (ENV_MATERIAL, SUBST_MATERIAL)))
    return dict(zip(uniq, make_nk_fn(uniq)))


def stop_if_no_s4(have_s4):
    if not have_s4:
        st.error("S4 (RCWA2D/libs4lab.so) を読み込めません。WSL上で起動し、未ビルドなら "
                 "`bash third_party/S4/build_s4lab.sh` を実行してください。")
        st.stop()


def fixed_action_bar_css(key, width_px):
    """st.container(key=key) を画面右下に固定表示するCSS（`.st-key-<key>` クラスに当てる）。"""
    st.html(f"""
    <style>
    .st-key-{key} {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        padding: 10px 14px 4px 14px;
        border-radius: 12px;
        box-shadow: 0 2px 14px rgba(0,0,0,0.35);
        background-color: #ffffff;
        border: 1px solid rgba(0,0,0,0.15);
        width: {width_px}px;
    }}
    @media (prefers-color-scheme: dark) {{
        .st-key-{key} {{
            background-color: #262730;
            border-color: rgba(255,255,255,0.15);
        }}
    }}
    </style>
    """)

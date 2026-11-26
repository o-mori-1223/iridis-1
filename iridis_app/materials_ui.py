"""サイドバーの「➕ 材料を追加」: n,k の表のファイルを data/nk に保存する（1D・2D 共通）。"""
import os

import streamlit as st

from RCWA2D.material_spec import parse_nk_text, save_nk

from .common import get_nk_list


def _decode(data):
    for enc in ('utf-8-sig', 'cp932'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    raise ValueError('文字コードを読めません（UTF-8 か Shift_JIS にしてください）。')


def render_add_material():
    flash = st.session_state.pop('nk_flash', None)
    with st.expander('➕ 材料を追加', expanded=bool(flash)):
        if flash:
            st.success(flash)
        n = st.session_state.get('nk_add_n', 0)   # 保存したら入力欄を新しくする
        up = st.file_uploader('n, k の表（1 列目 = 波長、2 列目 = n、3 列目 = k）', type=['nk', 'csv', 'txt', 'dat'], key=f'nk_add_file_{n}',
                              help='区切りは空白・タブ・カンマ。見出しの行は飛ばす。波長は nm（最大が 20 未満なら µm とみなす）。'
                                   'k の列が無ければ 0。吸収は k > 0 で表す。')
        default = ''.join(c if c.isascii() and (c.isalnum() or c in '_-') else '_' for c in os.path.splitext(up.name)[0])[:40] if up else ''
        name = st.text_input('材料の名前（半角英数字・_・-）', value=default, key=f'nk_add_name_{n}_{default}')
        source = st.text_input('出典（任意）', key=f'nk_add_source_{n}', placeholder='例: 分光エリプソメトリー 2026-11')
        if up is None:
            return
        try:
            d = parse_nk_text(_decode(up.getvalue()))
        except ValueError as e:
            st.error(f'読めません: {e}')
            return
        st.caption(f'{d[0, 0]:g}〜{d[-1, 0]:g} nm・{len(d)} 点・n {d[:, 1].min():.3g}〜{d[:, 1].max():.3g}・k {d[:, 2].min():.3g}〜{d[:, 2].max():.3g}')
        if d[0, 0] > 380 or d[-1, 0] < 780:
            st.warning('可視域（380〜780 nm）の一部がありません。無い波長は端の 2 点から直線で延ばした値になります。')
        if st.button('保存', key='nk_add_save', disabled=not name.strip()):
            try:
                save_nk(name, d, source)
            except (ValueError, OSError) as e:
                st.error(str(e))
                return
            get_nk_list.clear()
            st.session_state['nk_add_n'] = n + 1
            st.session_state['nk_flash'] = f'「{name.strip()}」を追加しました（data/nk/{name.strip()}.nk）。材料の一覧に出ます。'
            st.rerun()

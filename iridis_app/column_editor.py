"""1Dモードの列ビルダー（st.session_state.columns を編集するUI）。"""
import numpy as np
import streamlit as st

from RCWA2D.structure_builder import _col_total_height


def _new_col_id():
    st.session_state._ncol_id += 1
    return st.session_state._ncol_id


def _new_layer_id():
    st.session_state._nlayer_id += 1
    return st.session_state._nlayer_id


def _new_film_layer(ltype, nk_namelist, nk_idx_film, nk_idx_subst):
    if ltype == 'simple':
        return {'id': _new_layer_id(), 'type': 'simple',
                'material': nk_namelist[nk_idx_film], 'thickness': 100.0}
    if ltype == 'pair':
        return {'id': _new_layer_id(), 'type': 'pair',
                'mat_a': nk_namelist[nk_idx_film], 'th_a': 50.0,
                'mat_b': nk_namelist[nk_idx_subst], 'th_b': 50.0, 'n_pairs': 3}
    return {'id': _new_layer_id(), 'type': 'taper',
            'material': nk_namelist[nk_idx_film], 'thickness': 200.0,
            'frac_bottom': 1.0, 'frac_top': 0.5, 'n_slices': 10, 'alignment': 'Center'}


def _render_film_layer(fl, li, cid, nk_namelist, nk_idx_film):
    """膜層1個分の編集ウィジェット。削除ボタンが押されたら True を返す。"""
    lid = fl['id']
    ftype = fl.get('type', 'simple')
    removed = False

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
            removed = st.button('✕', key=f'lrm_{cid}_{lid}', help='この層を削除')

    elif ftype == 'pair':
        st.markdown(f'*🔄 ペア繰り返し層 #{li+1}*',
                    help='A = 上側（空気側）/ B = 下側（基板側）。ppap.py の "Layer A (top)" と同じ規約。')
        p1, p2, p3 = st.columns([2, 2, 1])
        with p1:
            ia_ = nk_namelist.index(fl['mat_a']) if fl.get('mat_a', '') in nk_namelist else nk_idx_film
            fl['mat_a'] = st.selectbox(f'材質A 上側 #{li+1}', nk_namelist, index=ia_, key=f'pma_{cid}_{lid}')
            fl['th_a']  = st.number_input(f'厚さA #{li+1} [nm]', 0.0, 1e6, fl.get('th_a', 50.0), 1.0, format='%g', key=f'pta_{cid}_{lid}')
        with p2:
            ib_ = nk_namelist.index(fl['mat_b']) if fl.get('mat_b', '') in nk_namelist else 0
            fl['mat_b'] = st.selectbox(f'材質B 下側 #{li+1}', nk_namelist, index=ib_, key=f'pmb_{cid}_{lid}')
            fl['th_b']  = st.number_input(f'厚さB #{li+1} [nm]', 0.0, 1e6, fl.get('th_b', 50.0), 1.0, format='%g', key=f'ptb_{cid}_{lid}')
        with p3:
            fl['n_pairs'] = st.number_input(f'繰り返し #{li+1}', 1, 200, fl.get('n_pairs', 3), 1, format='%d', key=f'pn_{cid}_{lid}')
            st.caption(f"合計 {fl['n_pairs'] * (fl['th_a'] + fl['th_b']):.0f} nm")
            removed = st.button('✕', key=f'lrm_{cid}_{lid}', help='この層を削除')

    elif ftype == 'taper':
        st.markdown(f'*📐 台形層 #{li+1}*')
        t1, t2, t3 = st.columns([2, 2, 2])
        with t1:
            it_ = nk_namelist.index(fl['material']) if fl.get('material', '') in nk_namelist else nk_idx_film
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
            removed = st.button('✕', key=f'lrm_{cid}_{lid}', help='この層を削除')

    st.divider()
    return removed


def render_column_builder(nk_namelist, nk_idx_film, nk_idx_subst, nk_subst_name):
    """列ビルダー全体。st.session_state.columns を直接書き換える
    （追加・削除・並べ替えのボタンが押された場合は st.rerun() する）。"""
    if 'columns' not in st.session_state:
        st.session_state.columns     = []
        st.session_state._ncol_id    = 0
        st.session_state._nlayer_id  = 0

    if st.button('＋ 列を追加 (Add column)',
                 help='周期を「列」に分割し、各列で基板の上げ高さとフィルム層を独立に設定する。'
                      '列の幅の合計がピッチになる。列は左から右の順。'):
        st.session_state.columns.append({
            'id': _new_col_id(), 'width': 300.0,
            'subst_raise': 0.0, 'raise_mode': 'sharp', 'film_layers': [],
        })

    _col_rm = _col_mv = _lrm = _ladd = None   # _ladd は (col_index, layer_type)

    for ci, col in enumerate(st.session_state.columns):
        cid     = col['id']
        total_h = _col_total_height(col)
        label   = f"Column {ci+1}  │  幅 {col['width']:.0f} nm  │  高さ {total_h:.0f} nm"
        # 列が多いと縦に長くなるので、2 列を超えたら最後に追加した列だけ開いておく
        with st.expander(label, expanded=len(st.session_state.columns) <= 2 or cid == st.session_state._ncol_id):

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
                    st.caption(f'▸ {nk_subst_name}: なだらか上げ ピーク {col["subst_raise"]:.0f} nm → 列端 '
                               f'≈{col["subst_raise"] / np.e:.0f} nm',
                               help=f'super-Gaussian の次数 n={_n:.2f}。列端から隣の列へなだらかに連続する。')
                else:
                    st.caption(f'▸ {nk_subst_name}: 0 → {col["subst_raise"]:.0f} nm')

            st.markdown('**フィルム層**（下から上）')
            for li, fl in enumerate(col['film_layers']):
                if _render_film_layer(fl, li, cid, nk_namelist, nk_idx_film):
                    _lrm = (ci, li)

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
        st.session_state.columns[ci]['film_layers'].append(
            _new_film_layer(ltype, nk_namelist, nk_idx_film, nk_idx_subst))
        st.rerun()
    if _col_rm is not None:
        st.session_state.columns.pop(_col_rm)
        st.rerun()
    if _col_mv is not None:
        ci, d = _col_mv; j = ci + d
        st.session_state.columns[ci], st.session_state.columns[j] = \
            st.session_state.columns[j], st.session_state.columns[ci]
        st.rerun()

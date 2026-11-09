"""2Dモードの編集UI: 層構成スタック（st.session_state.array_layers）と
基板段差（st.session_state.step_regions / step_film_layers）。"""
import numpy as np
import streamlit as st

from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL

from .array2d_model import hcylinder_footprint_nm, hcylinder_overlaps, layer_thickness_nm

_SUBST_SAME_LABEL = '基板と同じ'
_ENV_SAME_LABEL = '環境と同じ'
_TYPE_LABEL = {'film': '平坦膜', 'disk': '円盤', 'sphere': '球', 'hemisphere': '半球',
               'hcylinder': '横倒しシリンダー', 'pair': 'ペア繰り返し'}


def _default(nk_namelist, name):
    return name if name in nk_namelist else nk_namelist[0]


def material_selectbox(nk_namelist, label, cur_value, key, allow_env_same=False, help=None):
    """材質選択。nk_namelist の材質に加えて「基板と同じ」(常時)・「環境と同じ」(背景材質用)
    を選べ、選ばれた場合は SUBST_MATERIAL / ENV_MATERIAL 定数を返す（backend がこれらを
    常に nk_subst_fn / nk_env_fn で扱うので、nk_fn_map への登録は不要）。"""
    extra = ([(_ENV_SAME_LABEL, ENV_MATERIAL)] if allow_env_same else []) + [(_SUBST_SAME_LABEL, SUBST_MATERIAL)]
    options = [lbl for lbl, _ in extra] + nk_namelist
    cur_label = next((lbl for lbl, sentinel in extra if sentinel == cur_value), cur_value)
    sel = st.selectbox(label, options, index=options.index(cur_label) if cur_label in options else 0,
                       key=key, help=help)
    return next((sentinel for lbl, sentinel in extra if sel == lbl), sel)


def _radius_warning(radius_nm, nn_dist_nm):
    if radius_nm >= nn_dist_nm / 2.0:
        st.warning(f"半径が最近接粒子間距離の半分（{nn_dist_nm / 2.0:.0f}nm）以上です。隣の周期像と重なります。")


def _position_inputs(spec, key_prefix, help):
    pc1, pc2 = st.columns(2)
    with pc1:
        spec['cx_nm'] = st.number_input('位置 x [nm]', -1e6, 1e6, spec.get('cx_nm', 0.0), 1.0,
                                        key=f'{key_prefix}_cx', help=help)
    with pc2:
        spec['cy_nm'] = st.number_input('位置 y [nm]', -1e6, 1e6, spec.get('cy_nm', 0.0), 1.0,
                                        key=f'{key_prefix}_cy')


def _render_sub_layer_fields(sub, key_prefix, nk_namelist, nn_dist_nm):
    """ペア繰り返しの A/B（film/disk）1個分の編集ウィジェット。`sub` を直接書き換える。"""
    sub_type_opts = ['平坦膜', '円盤']
    sel = st.radio('種類', sub_type_opts, horizontal=True,
                   index=1 if sub.get('type') == 'disk' else 0, key=f'{key_prefix}_type')
    sub['type'] = 'disk' if sel == '円盤' else 'film'
    if sub['type'] == 'film':
        fc1, fc2 = st.columns(2)
        with fc1:
            sub['material'] = material_selectbox(nk_namelist, '材質', sub.get('material', ''), f'{key_prefix}_mat')
        with fc2:
            sub['thickness_nm'] = st.number_input(
                '厚さ [nm]', 1.0, 5000.0, sub.get('thickness_nm', 100.0), 1.0, key=f'{key_prefix}_th')
        return
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        sub['material'] = material_selectbox(nk_namelist, '粒子材質', sub.get('material', ''), f'{key_prefix}_mat')
    with gc2:
        sub['bg_material'] = material_selectbox(
            nk_namelist, '背景材質', sub.get('bg_material', ENV_MATERIAL), f'{key_prefix}_bg', allow_env_same=True)
    with gc3:
        sub['radius_nm'] = st.number_input(
            '半径 [nm]', 5.0, 2000.0, sub.get('radius_nm', 150.0), 5.0, key=f'{key_prefix}_r')
    sub['thickness_nm'] = st.number_input(
        '高さ [nm]', 1.0, 2000.0, sub.get('thickness_nm', 100.0), 1.0, key=f'{key_prefix}_th')
    _position_inputs(sub, key_prefix,
                     help='格子原点からのオフセット。既定の(0,0)は格子点（通常の粒子位置）。'
                          '基板材質の台座を粒子と別位置に置きたい場合などにずらす。')
    _radius_warning(sub['radius_nm'], nn_dist_nm)


def _render_hcylinder(lyr, nk_namelist, a1, a2):
    """横倒しシリンダー（軸が x/y 方向、高さ = 直径）の編集ウィジェット。"""
    lid = lyr['id']
    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        lyr['material'] = material_selectbox(nk_namelist, '材質', lyr.get('material', ''), f'al_mat_{lid}')
    with gc2:
        lyr['bg_material'] = material_selectbox(
            nk_namelist, 'この層の背景材質', lyr.get('bg_material', ENV_MATERIAL), f'al_bg_{lid}',
            allow_env_same=True, help='シリンダーの周り（同じ高さでシリンダーが占めない領域）の材質。')
    with gc3:
        lyr['radius_nm'] = st.number_input('半径 [nm]', 5.0, 2000.0, lyr.get('radius_nm', 100.0), 5.0,
                                           key=f'al_r_{lid}')
    rect_lattice = abs(float(a2[0])) < 1e-6
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        axis = st.radio('軸の向き', ['x 方向', 'y 方向'], horizontal=True,
                        index=1 if lyr.get('axis') == 'y' else 0, key=f'al_axis_{lid}')
        lyr['axis'] = 'y' if axis.startswith('y') else 'x'
    with hc2:
        can_inf = lyr['axis'] == 'x' or rect_lattice
        lyr['infinite'] = st.checkbox(
            '周期いっぱい（無限に長い棒）', value=bool(lyr.get('infinite')) and can_inf, key=f'al_inf_{lid}',
            disabled=not can_inf,
            help='長さを軸方向の周期と同じにし、端を隣の周期像とつなげる（1次元の格子と同じ）。'
                 + ('' if can_inf else ' y 方向は格子の角度が 90° のときだけ使える。'))
    with hc3:
        lyr['length_nm'] = st.number_input('長さ [nm]', 5.0, 1e5, lyr.get('length_nm', 400.0), 5.0,
                                           key=f'al_len_{lid}', disabled=lyr['infinite'])
    lyr['thickness_nm'] = layer_thickness_nm(lyr)
    lyr['n_slices'] = st.number_input(
        'staircase分割数 n_slices', 2, 200, lyr.get('n_slices', 20), 2, key=f'al_ns_{lid}',
        help='断面の円を z 方向に n_slices 枚の薄い長方形で近似する（球と同じ中点則）。'
             '20前後から始め、増やして結果が変わらないか確かめるとよい。')
    hx, hy = hcylinder_footprint_nm(lyr, a1, a2)
    st.caption(f"高さ（直径） = {lyr['thickness_nm']:.0f} nm。真上から見た外形 {2 * hx:.0f} × {2 * hy:.0f} nm"
               f"（{'x' if lyr['axis'] == 'x' else 'y'} 方向が軸）。端面は平ら。")
    pc1, pc2 = st.columns(2)
    with pc1:
        lyr['cx_nm'] = st.number_input('位置 x [nm]', -1e6, 1e6, lyr.get('cx_nm', 0.0), 1.0, key=f'al_cx_{lid}',
                                       help='格子原点からのオフセット（軸の中心の位置）。')
    with pc2:
        lyr['cy_nm'] = st.number_input('位置 y [nm]', -1e6, 1e6, lyr.get('cy_nm', 0.0), 1.0, key=f'al_cy_{lid}')
    if hcylinder_overlaps(lyr, a1, a2):
        st.error("シリンダーが隣の周期像と重なります（直径または長さが周期より大きい）。重なった構造は正しく"
                 "計算できません。半径・長さを小さくするか周期を広げてください（端をつなげたい場合は"
                 "「周期いっぱい」を使う）。")


def _render_layer(lyr, nk_namelist, nn_dist_nm, a1, a2):
    lid = lyr['id']
    if lyr['type'] == 'hcylinder':
        _render_hcylinder(lyr, nk_namelist, a1, a2)
    elif lyr['type'] == 'film':
        fc1, fc2 = st.columns(2)
        with fc1:
            lyr['material'] = material_selectbox(nk_namelist, '材質', lyr.get('material', ''), f'al_mat_{lid}')
        with fc2:
            lyr['thickness_nm'] = st.number_input(
                '厚さ [nm]', 1.0, 5000.0, lyr.get('thickness_nm', 100.0), 1.0, key=f'al_th_{lid}')
    elif lyr['type'] == 'pair':
        st.markdown('**A（上側）**')
        _render_sub_layer_fields(lyr['a'], f'al_{lid}_a', nk_namelist, nn_dist_nm)
        st.markdown('**B（下側）**')
        _render_sub_layer_fields(lyr['b'], f'al_{lid}_b', nk_namelist, nn_dist_nm)
        lyr['n_pairs'] = st.number_input(
            '繰り返し回数 n_pairs', 1, 200, lyr.get('n_pairs', 3), 1, key=f'al_np_{lid}')
        pair_th = layer_thickness_nm(lyr['a']) + layer_thickness_nm(lyr['b'])
        st.caption(f"1ペアの厚さ = {pair_th:.0f} nm × {int(lyr['n_pairs'])}回"
                   f" = 合計 {pair_th * lyr['n_pairs']:.0f} nm")
    else:  # disk / sphere
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            lyr['material'] = material_selectbox(nk_namelist, '粒子材質', lyr.get('material', ''), f'al_mat_{lid}')
        with gc2:
            lyr['bg_material'] = material_selectbox(
                nk_namelist, 'この層の背景材質', lyr.get('bg_material', ENV_MATERIAL), f'al_bg_{lid}',
                allow_env_same=True,
                help='粒子の周囲（同じ層内で粒子が占めない領域）の材質。'
                     '「環境と同じ」なら一番上の環境設定と同じ屈折率を使う。')
        with gc3:
            lyr['radius_nm'] = st.number_input(
                '半径 [nm]', 5.0, 2000.0, lyr.get('radius_nm', 150.0), 5.0, key=f'al_r_{lid}')
        if lyr['type'] == 'disk':
            lyr['thickness_nm'] = st.number_input(
                '高さ [nm]', 1.0, 2000.0, lyr.get('thickness_nm', 100.0), 1.0, key=f'al_th_{lid}')
        else:
            lyr['thickness_nm'] = layer_thickness_nm(lyr)
            if lyr['type'] == 'sphere':
                st.caption(f"球の高さ（直径） = {lyr['thickness_nm']:.0f} nm（半径から自動計算）")
            else:
                st.caption(f"半球の高さ = 半径 = {lyr['thickness_nm']:.0f} nm"
                           "（平らな面を下＝基板側にしたドーム形）")
            lyr['n_slices'] = st.number_input(
                'staircase分割数 n_slices', 2, 200, lyr.get('n_slices', 20), 2, key=f'al_ns_{lid}',
                help='球/半球をz方向にn_slices枚の薄い円盤で近似する。目安: 20前後で概ね収束、'
                     '精密な数値が必要なら40-80程度まで増やして収束を確認すること。')
        pc1, pc2 = st.columns(2)
        with pc1:
            lyr['cx_nm'] = st.number_input(
                '位置 x [nm]', -1e6, 1e6, lyr.get('cx_nm', 0.0), 1.0, key=f'al_cx_{lid}',
                help='格子原点からのオフセット。既定の(0,0)は格子点（通常の粒子位置）。'
                     '基板材質を選んで台座(ポスト)として使う場合などにずらす。')
        with pc2:
            lyr['cy_nm'] = st.number_input(
                '位置 y [nm]', -1e6, 1e6, lyr.get('cy_nm', 0.0), 1.0, key=f'al_cy_{lid}')
        _radius_warning(lyr['radius_nm'], nn_dist_nm)


def _new_array_layer(kind, nk_namelist):
    st.session_state._array_layer_id += 1
    new = {'id': st.session_state._array_layer_id, 'type': kind}
    mat_a, mat_b = _default(nk_namelist, 'TiO2'), _default(nk_namelist, 'SiO2')
    if kind == 'film':
        new.update(material=nk_namelist[0], thickness_nm=100.0)
    elif kind == 'disk':
        new.update(material=mat_a, bg_material=ENV_MATERIAL, thickness_nm=100.0, radius_nm=150.0)
    elif kind in ('sphere', 'hemisphere'):
        new.update(material=mat_a, bg_material=ENV_MATERIAL, radius_nm=150.0, n_slices=20)
    elif kind == 'hcylinder':
        new.update(material=mat_a, bg_material=ENV_MATERIAL, radius_nm=100.0, length_nm=400.0, axis='x',
                   infinite=False, n_slices=20, thickness_nm=200.0)
    else:  # pair
        new.update(n_pairs=3,
                   a={'type': 'disk', 'material': mat_a, 'bg_material': ENV_MATERIAL,
                      'thickness_nm': 100.0, 'radius_nm': 150.0},
                   b={'type': 'film', 'material': mat_b, 'thickness_nm': 80.0})
    return new


def render_layer_stack_editor(nk_namelist, nn_dist_nm, a1, a2):
    """層構成（垂直方向のスタック、top-first）。平坦膜・円盤・球・半球・横倒しシリンダー・
    ペア繰り返しを任意の順で積む。"""

    if 'array_layers' not in st.session_state:
        st.session_state.array_layers = [{
            'id': 1, 'type': 'disk', 'material': _default(nk_namelist, 'TiO2'),
            'bg_material': ENV_MATERIAL, 'thickness_nm': 100.0, 'radius_nm': 150.0, 'n_slices': 20,
        }]
        st.session_state._array_layer_id = 1

    _al_add = _al_rm = _al_mv = None
    for col, (label, kind) in zip(st.columns(6), (('＋ 平坦膜を追加', 'film'), ('＋ 円盤層を追加', 'disk'),
                                                   ('＋ 球層を追加', 'sphere'), ('＋ 半球層を追加', 'hemisphere'),
                                                   ('＋ 横倒しシリンダー', 'hcylinder'),
                                                   ('＋ ペア繰り返しを追加', 'pair'))):
        with col:
            if st.button(label):
                _al_add = kind

    layers = st.session_state.array_layers
    for li, lyr in enumerate(layers):
        lid = lyr['id']
        # 層が多いと縦に長くなるので、3 層を超えたら最後に追加した層だけ開いておく
        with st.expander(f"層 {li+1}（上から） — {_TYPE_LABEL[lyr['type']]}",
                         expanded=len(layers) <= 3 or lid == st.session_state._array_layer_id):
            _render_layer(lyr, nk_namelist, nn_dist_nm, a1, a2)
            mvc1, mvc2, mvc3 = st.columns(3)
            with mvc1:
                if st.button('▲ 上へ', key=f'al_up_{lid}', disabled=(li == 0)):
                    _al_mv = (li, -1)
            with mvc2:
                if st.button('▼ 下へ', key=f'al_dn_{lid}', disabled=(li == len(layers) - 1)):
                    _al_mv = (li, 1)
            with mvc3:
                if st.button('✕ この層を削除', key=f'al_rm_{lid}'):
                    _al_rm = li

    if _al_add is not None:
        layers.append(_new_array_layer(_al_add, nk_namelist))
        st.rerun()
    if _al_rm is not None:
        layers.pop(_al_rm)
        st.rerun()
    if _al_mv is not None:
        li, mv = _al_mv
        layers[li], layers[li + mv] = layers[li + mv], layers[li]
        st.rerun()

    if not layers:
        st.info('「＋」ボタンで層を追加してください。')


def _bounding_r(reg):
    """重なり警告用の簡易バウンディング半径（矩形は対角線の半分）。regions_to_layers_2d は
    重なった領域を正しく扱えないので、「重なっているかもしれない」ことを知らせるのが目的。"""
    if reg['shape'] == 'circle':
        return reg.get('radius_nm', 0.0)
    return (reg.get('hw_x_nm', 0.0) ** 2 + reg.get('hw_y_nm', 0.0) ** 2) ** 0.5


def _new_step_region():
    st.session_state._step_region_id += 1
    return {'id': st.session_state._step_region_id, 'shape': 'rect', 'cx_nm': 0.0, 'cy_nm': 0.0,
            'hw_x_nm': 150.0, 'hw_y_nm': 150.0, 'radius_nm': 150.0, 'raise_nm': 50.0}


def _warn_region_exceeds_cell(ri, reg, a1, a2):
    """上げ領域が単位胞より大きいと、隣の周期像と重なって非物理的な結果（R_total > 1 など。
    実測: ピッチ400nmに300×900nmの矩形で R_total=1.20）になるので警告する。
    単位胞ぴったり（領域の幅 = 周期）は可。"""
    cell_w = float(np.linalg.norm(a1))
    cell_h = abs(float(a1[0] * a2[1] - a1[1] * a2[0])) / cell_w   # a1 に垂直な方向の周期
    if reg['shape'] == 'circle':
        w = h = 2.0 * reg.get('radius_nm', 0.0)
    else:
        w, h = 2.0 * reg.get('hw_x_nm', 0.0), 2.0 * reg.get('hw_y_nm', 0.0)
    if w > cell_w + 1e-6 or h > cell_h + 1e-6:
        st.error(f"領域{ri+1}（{w:.0f}×{h:.0f} nm）が単位胞（x方向 {cell_w:.0f} nm、y方向 {cell_h:.0f} nm）"
                 "より大きく、隣の周期像と重なります。この構造は正しく計算できません"
                 "（反射率が1を超えるなど非物理的な結果になります）。ピッチを広げるか領域を小さくしてください。")


def _render_step_regions(a1, a2):
    st.markdown("**上げ領域**", help="基板を上げる円/矩形の位置・サイズ・高さ。")
    regions = st.session_state.step_regions
    _add = st.button("＋ 上げ領域を追加", key="step_region_add")
    _rm = None
    for ri, reg in enumerate(regions):
        rid = reg['id']
        with st.expander(f"領域 {ri+1}", expanded=True):
            shape_sel = st.radio('形状', ['矩形', '円'], horizontal=True,
                                 index=1 if reg.get('shape') == 'circle' else 0, key=f'step_shape_{rid}')
            reg['shape'] = 'circle' if shape_sel == '円' else 'rect'
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                reg['cx_nm'] = st.number_input('位置 x [nm]', -1e6, 1e6, reg.get('cx_nm', 0.0), 1.0, key=f'step_cx_{rid}')
            with rc2:
                reg['cy_nm'] = st.number_input('位置 y [nm]', -1e6, 1e6, reg.get('cy_nm', 0.0), 1.0, key=f'step_cy_{rid}')
            with rc3:
                reg['raise_nm'] = st.number_input('上げ高さ [nm]', 0.0, 1e6, reg.get('raise_nm', 50.0), 1.0, key=f'step_raise_{rid}')
            if reg['shape'] == 'rect':
                sc1, sc2 = st.columns(2)
                with sc1:
                    reg['hw_x_nm'] = st.number_input('半幅 x [nm]', 1.0, 5000.0, reg.get('hw_x_nm', 150.0), 1.0, key=f'step_hwx_{rid}')
                with sc2:
                    reg['hw_y_nm'] = st.number_input('半幅 y [nm]', 1.0, 5000.0, reg.get('hw_y_nm', 150.0), 1.0, key=f'step_hwy_{rid}')
                st.caption(f"一辺 {reg['hw_x_nm']*2:.0f}×{reg['hw_y_nm']*2:.0f} nm")
            else:
                reg['radius_nm'] = st.number_input('半径 [nm]', 1.0, 5000.0, reg.get('radius_nm', 150.0), 1.0, key=f'step_rad_{rid}')
            if st.button('✕ この領域を削除', key=f'step_rm_{rid}'):
                _rm = ri

    if _add:
        regions.append(_new_step_region())
        st.rerun()
    if _rm is not None:
        regions.pop(_rm)
        st.rerun()

    for i, reg in enumerate(regions):
        _warn_region_exceeds_cell(i, reg, a1, a2)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            dx = regions[i]['cx_nm'] - regions[j]['cx_nm']
            dy = regions[i]['cy_nm'] - regions[j]['cy_nm']
            if (dx ** 2 + dy ** 2) ** 0.5 < _bounding_r(regions[i]) + _bounding_r(regions[j]):
                st.warning(f"領域{i+1}と領域{j+1}が重なっている可能性があります。"
                           "上げ領域同士が重なる構造は正しく扱えません。")


def render_film_recipe(nk_namelist, list_key, id_key, prefix, heading, empty_message, heading_help=None):
    """成膜レシピ（単層・ペア繰り返し、下から上）の編集UI。st.session_state[list_key] を直接
    書き換える。基板段差（prefix='step'）とレンズ状の凹凸（prefix='lens'）で共用する。"""
    st.markdown(heading, help=heading_help)
    films = st.session_state[list_key]
    _add = _rm = None
    sfc1, sfc2 = st.columns(2)
    with sfc1:
        if st.button("＋ 単層を追加", key=f"{prefix}_film_add_simple"):
            _add = 'simple'
    with sfc2:
        if st.button("＋ ペア繰り返しを追加", key=f"{prefix}_film_add_pair"):
            _add = 'pair'

    for fi, fl in enumerate(films):
        flid = fl['id']
        if fl.get('type', 'simple') == 'simple':
            lc1, lc2, lc3 = st.columns([3, 2, 1])
            with lc1:
                fl['material'] = material_selectbox(nk_namelist, f'材質 #{fi+1}', fl.get('material', ''), f'{prefix}_fm_{flid}')
            with lc2:
                fl['thickness'] = st.number_input(
                    f'厚さ #{fi+1} [nm]', 0.0, 1e6, fl.get('thickness', 100.0), 1.0, key=f'{prefix}_ft_{flid}')
            with lc3:
                st.write('')
                if st.button('✕', key=f'{prefix}_frm_{flid}', help='この層を削除'):
                    _rm = fi
        else:  # pair（A = 上側＝環境側、B = 下側＝基板側。ラベルに書いてある）
            p1, p2, p3 = st.columns([2, 2, 1])
            with p1:
                fl['mat_a'] = material_selectbox(nk_namelist, f'材質A 上側 #{fi+1}', fl.get('mat_a', ''), f'{prefix}_pma_{flid}')
                fl['th_a'] = st.number_input(f'厚さA #{fi+1} [nm]', 0.0, 1e6, fl.get('th_a', 50.0), 1.0, key=f'{prefix}_pta_{flid}')
            with p2:
                fl['mat_b'] = material_selectbox(nk_namelist, f'材質B 下側 #{fi+1}', fl.get('mat_b', ''), f'{prefix}_pmb_{flid}')
                fl['th_b'] = st.number_input(f'厚さB #{fi+1} [nm]', 0.0, 1e6, fl.get('th_b', 50.0), 1.0, key=f'{prefix}_ptb_{flid}')
            with p3:
                fl['n_pairs'] = st.number_input(f'繰り返し #{fi+1}', 1, 200, fl.get('n_pairs', 3), 1, key=f'{prefix}_pn_{flid}')
                st.caption(f"合計 {fl['n_pairs']*(fl['th_a']+fl['th_b']):.0f} nm")
                if st.button('✕', key=f'{prefix}_frm_{flid}', help='この層を削除'):
                    _rm = fi
        st.divider()

    if _add is not None:
        st.session_state[id_key] += 1
        lid = st.session_state[id_key]
        mat_a, mat_b = _default(nk_namelist, 'TiO2'), _default(nk_namelist, 'SiO2')
        films.append({'id': lid, 'type': 'simple', 'material': mat_a, 'thickness': 100.0} if _add == 'simple' else
                     {'id': lid, 'type': 'pair', 'mat_a': mat_a, 'th_a': 51.0, 'mat_b': mat_b, 'th_b': 81.0,
                      'n_pairs': 3})
        st.rerun()
    if _rm is not None:
        films.pop(_rm)
        st.rerun()

    if not films:
        st.info(empty_message)


def _render_step_film(nk_namelist):
    render_film_recipe(nk_namelist, 'step_film_layers', '_step_layer_id', 'step', "**成膜レシピ**（下から上）",
                       "成膜レシピに層を追加してください（空のままだと、上げ領域は基板材質の台座のみになります）。",
                       heading_help="上げ領域とフィールド（周り）の両方に、下から上の順に積む。")


def render_step_editor(nk_namelist, a1, a2):
    """基板段差（コンフォーマル多層、任意）。1Dの列ごとの基板上げを2Dに拡張したもの:
    基板の一部領域(円/矩形)だけを上げ、周囲(フィールド)と上げ領域の両方に同じ成膜レシピを
    積む（structure_builder.regions_to_layers_2d がバンド展開する）。デフォルトOFF。
    有効なら step_regions を返す（無効・領域なしなら None）。"""
    enabled = st.checkbox(
        "基板の一部領域だけを上げ、その上に下の「成膜レシピ」を段差に追従させて積む",
        value=False, key="step_enabled",
        help="基板の一部（円または矩形の領域）だけを指定高さ分だけ上げ、周囲"
             "（フィールド）と上げ領域の両方に同じ成膜レシピを積むことで、段差に"
             "追従した多層膜（例: DBRミラーの一部だけが盛り上がった構造）を表現"
             "する。offのときは既存の「層構成」の挙動に一切影響しない。"
             "段差を足すと S4 に渡す層が大きく増え（上げ領域の数・成膜レシピの層数にほぼ比例）、計算が"
             "大幅に遅くなることがあるので、「⏱ 見積もり・収束確認」タブで 1 点あたりの時間を確かめるとよい。")
    if not enabled:
        return None

    if 'step_regions' not in st.session_state:
        st.session_state.step_regions = [{'id': 1, 'shape': 'rect', 'cx_nm': 0.0, 'cy_nm': 0.0,
                                          'hw_x_nm': 150.0, 'hw_y_nm': 150.0, 'radius_nm': 150.0,
                                          'raise_nm': 50.0}]
        st.session_state._step_region_id = 1
    if 'step_film_layers' not in st.session_state:
        st.session_state.step_film_layers = []
        st.session_state._step_layer_id = 0

    _render_step_regions(a1, a2)
    _render_step_film(nk_namelist)
    return st.session_state.step_regions or None

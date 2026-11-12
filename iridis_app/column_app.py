"""1Dモード（列ベース構造）の画面。旧 R8_koumyou_v4.py を関数化したもの。"""
import copy
import datetime
import time
import uuid

import colour
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from RCWA2D.color_display import render_gsolver_style_display
from RCWA2D.gsolver_export import layers_to_gsolver_text, DEFAULT_NAME_MAP
from RCWA2D.rcwa_s4_backend import HAVE_S4
from RCWA2D.structure_builder import (
    ENV_MATERIAL, SUBST_MATERIAL, columns_to_layers, expand_graded_columns,
)

from .column_calc import calc_1d, layers_cache_key
from .column_editor import render_column_builder
from .column_figures import create_structure_figure, create_crosssection_fig, get_structure_breakdown_df
from .common import (TAB_PREVIEW, TAB_RESULT, TAB_STRUCT, build_nk_fn_map, convert_df, fixed_action_bar_css,
                     get_nk_list, main_tabs, make_nk_fn, show_result_tab, stop_if_no_s4)
from .records import to_jsonable
from .records_ui import consume_autorun, render_restore_notice, render_save_panel

_SPECTRUM_MENU = ('Visible[380-780nm]', 'UV[200-400nm]', 'NIR[700-1000nm]', 'All[200-1000nm]', 'Any')
_SPECTRUM_PRESET = {_SPECTRUM_MENU[0]: (380.0, 780.0, 81), _SPECTRUM_MENU[1]: (200.0, 400.0, 101),
                    _SPECTRUM_MENU[2]: (700.0, 1000.0, 61), _SPECTRUM_MENU[3]: (200.0, 1000.0, 81)}
# サイドバーの表示名（値そのものは解析記録との互換のため英語のまま）
_SPECTRUM_LABEL = dict(zip(_SPECTRUM_MENU, ('可視 380〜780 nm', '紫外 200〜400 nm', '近赤外 700〜1000 nm',
                                            '全域 200〜1000 nm', '任意（範囲を指定）')))
_ANGLE_MODE_LABEL = {'Single angle (2D)': '単一角度', 'Angle sweep (3D)': '角度掃引'}
_SWATCH_THETAS = [0.0, 30.0, 60.0]


def _sd_to_hex(arr, wl_nm_ar, name):
    sd = colour.SpectralDistribution(arr, name=name)
    sd.wavelengths = wl_nm_ar
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    ill  = colour.SDS_ILLUMINANTS['D65']
    XYZ  = colour.sd_to_XYZ(sd, cmfs, ill)
    RGB  = colour.XYZ_to_sRGB(XYZ / 100)
    b    = [int(np.clip(round(v*255), 0, 255)) for v in RGB]
    return XYZ, '#{:02x}{:02x}{:02x}'.format(*b)


def _structure_figures(fi):
    """解析記録に添える構造の図（膜構成の図と断面図）。fi は計算した時点の写し。"""
    return {
        'structure': create_structure_figure(fi['layers'], fi['pitch_nm'], fi['subst'], col_widths=fi['col_widths']),
        'cross_section': create_crosssection_fig(fi['layers'], fi['pitch_nm'], col_widths=fi['col_widths'], n_repeats=2),
    }


def _film_material_names(columns):
    names = []
    for col in columns:
        for fl in col['film_layers']:
            ft = fl.get('type', 'simple')
            if ft in ('simple', 'taper'):
                names.append(fl.get('material', ''))
            elif ft == 'pair':
                names += [fl.get('mat_a', ''), fl.get('mat_b', '')]
    return [m for m in names if m and m not in (ENV_MATERIAL, SUBST_MATERIAL)]


def render_column_app():
    st.title('Iridis — 列ベース構造 (1D)')
    render_restore_notice('1d')
    stop_if_no_s4(HAVE_S4)

    nk_namelist = get_nk_list()
    nk_idx_subst = nk_namelist.index('Silicon')
    nk_idx_film  = nk_namelist.index('SiO2')

    # ================================================================
    # Sidebar
    # ================================================================
    # サイドバーは縦に長くならないよう 2 つのタブに分ける。見出しは 2D（array2d_app._sidebar）と揃える。
    # 選択肢の値（'Single angle (2D)' など）は解析記録に保存されているので変えず、表示だけ format_func で日本語にする。
    sb_med, sb_light = st.sidebar.tabs(['環境・基板', '入射・波長'])
    with sb_light:
        angle_mode = st.radio('入射角モード', ('Single angle (2D)', 'Angle sweep (3D)'), horizontal=True,
                              format_func=_ANGLE_MODE_LABEL.get, key='angle_mode')
        # key は解析記録からの読み込み（records_restore.restore_1d）が書き戻す先
        if angle_mode == 'Single angle (2D)':
            inc_angle = st.number_input('入射角 θ [deg]', 0.0, 89.0, 0.0, 0.1, format='%3.1f', key='inc_angle')
        else:
            angle_range = st.slider('入射角 θ の範囲 [deg]', 0.0, 89.0, (0.0, 80.0), 1.0, format='%.0f',
                                    key='angle_range')
            angle_step  = st.number_input('角度ステップ [deg]', 0.5, 45.0, 2.0, 0.5, format='%.1f', key='angle_step')
            inc_angle   = angle_range[0]

        st.divider()
        wl_option = st.selectbox('波長範囲', _SPECTRUM_MENU, format_func=_SPECTRUM_LABEL.get, key='wl_option')
        wl_min, wl_max, wl_n = _SPECTRUM_PRESET.get(wl_option, (400.0, 800.0, 81))
        if wl_option == 'Any':
            wl_min, wl_max = st.slider('波長 min〜max [nm]', 200.0, 1000.0, (wl_min, wl_max), 20.0, format='%.0f',
                                       key='wl_range')
            wl_n = st.number_input('波長点数', 11, 101, int(wl_n), 10, format='%d', key='wln')
        norder = st.number_input('NumBasis（次数）', min_value=5, max_value=101, value=11, step=2, format='%d',
                                 key='nOrder', help='大きいほど正確だが、計算時間は増える。')

    with sb_med:
        n_env = st.number_input('環境 n（構造の上の媒質）', 1.0, 3.0, 1.0, 0.01, format='%3.2f', key='nenv',
                                help='空気・真空は 1.00。')
        st.divider()
        nk_subst_name = st.selectbox('基板の材質', nk_namelist, index=nk_idx_subst, key='substrate')
        _custom_subst = st.checkbox(
            'カスタム数値（基板 n, k を直接入力）',
            value=False, key='custom_subst_nk',
            help='Gsolverなど他ツールの基板光学定数を直接指定して比較する場合に使用')
        if _custom_subst:
            _sc1, _sc2 = st.columns(2)
            with _sc1:
                _subst_n = st.number_input('基板 n', 0.0, 20.0,
                    st.session_state.get('subst_n_custom', 3.0), 0.01, format='%5.3f', key='subst_n_custom')
            with _sc2:
                _subst_k = st.number_input('基板 k', 0.0, 20.0,
                    st.session_state.get('subst_k_custom', 4.0), 0.01, format='%5.3f', key='subst_k_custom',
                    help='n, k は全波長で同じ定数として使う。')

    t_struct, t_prev, t_res = main_tabs('1d', [TAB_STRUCT, TAB_PREVIEW, TAB_RESULT])

    # ================================================================
    # Column builder → RCWA layers
    # ================================================================
    with t_struct:
        render_column_builder(nk_namelist, nk_idx_film, nk_idx_subst, nk_subst_name)

    columns    = st.session_state.columns
    col_widths = [c['width'] for c in columns]
    pitch_nm   = sum(col_widths) if col_widths else 500.0
    pitch_um   = pitch_nm / 1000.0

    with t_struct:
        if any(c.get('raise_mode') == 'graded' for c in columns):
            graded_n_slices = st.number_input(
                'なだらか基板プロファイルの分割数 (列ごと)', 2, 100, 10, 1, format='%d',
                key='graded_n_slices',
                help='graded設定の列を、この本数だけ細い子列に分割してsuper-Gaussian'
                     'プロファイル（および隣接列への裾の侵入）を階段状に近似します。'
                     '多いほど滑らかですが、RCWA解析層数がおおむね比例して増え、'
                     '計算が大きく遅くなります。')
            layers = columns_to_layers(expand_graded_columns(columns, int(graded_n_slices)))
        else:
            layers = columns_to_layers(columns)

        if columns:
            st.caption(f'ピッチ {pitch_nm:.1f} nm ／ RCWA 解析層数 {len(layers)} 層',
                       help='モデル: 等形成膜（列ごとに膜を積む）。')
        else:
            st.info('「列を追加」ボタンで列を追加してください（空の場合は平坦基板として計算されます）。')

        # ================================================================
        # Gsolver export
        # ================================================================
        with st.expander('📤 Gsolverインポート用テキストを生成', expanded=False):
            st.caption(
                'GSolver の File > Import Text 用。使う前に「？」の注意を確認', help=(
                    '材質名は GSolver.ini の [CONSTANTS] カタログに同名で登録されている必要がある'
                    f'（基板は「{nk_subst_name}」という名前で登録されている前提。カスタム n+ik の値は反映されないので、'
                    '先に GSolver.ini へ登録して名前を揃える）。インポート後は GSolver 側の断面プレビューで形を確認し、'
                    'Approximate ボタンは押さない（寸法が Canvas Grid 精度に再量子化される）。'
                    'ORDERS は 43 以上を推奨。'))
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
    # Structure figures（左に膜構成、右に周期構造の断面。開いていないときは図を描かない）
    # ================================================================
    with t_prev:
        _rep_c1, _rep_c2 = st.columns([4, 1])
        with _rep_c2:
            _n_rep = st.number_input('表示周期数', 1, 6, 2, 1, key='crosssec_nrep',
                                     help='右の断面図に並べる周期の数。横軸 = x 位置 [nm]（1周期 = ピッチ）、'
                                          '縦軸 = 高さ [nm]。等形成膜では高さによって列ごとに違う材料が現れる。')
        if t_prev.open:
            _fig_l, _fig_r = st.columns(2)
            with _fig_l:
                st.plotly_chart(create_structure_figure(layers, pitch_nm, nk_subst_name, col_widths=col_widths),
                                width="stretch")
            with _fig_r:
                if layers:
                    st.plotly_chart(create_crosssection_fig(layers, pitch_nm, col_widths=col_widths,
                                                            n_repeats=int(_n_rep)), width="stretch")
                else:
                    st.info('列を追加すると断面図が表示されます。')

        if columns:
            with st.expander('層テーブル — 各高さでの材料と層の種類', expanded=False):
                df_struct = get_structure_breakdown_df(columns)

                def _highlight_lattice(row):
                    if '格子層' in str(row.get('層の種類', '')):
                        return ['background-color: #fff3cd'] * len(row)
                    return [''] * len(row)
                st.dataframe(df_struct.style.apply(_highlight_lattice, axis=1),
                             width="stretch", hide_index=True, height=320)

    # ---- 計算コントロールは画面右下に固定。どのタブを開いていても見えるよう、タブの外に置く ----
    fixed_action_bar_css('calc_action_bar', 260)
    bar = st.container(key='calc_action_bar')

    # ================================================================
    # Spectrum / RCWA
    # ================================================================
    with t_res:
        nk_fn_map = build_nk_fn_map(_film_material_names(columns))
        if st.session_state.get('custom_subst_nk', False):
            nk_subst_fn = make_nk_fn([complex(st.session_state.get('subst_n_custom', 3.0),
                                              st.session_state.get('subst_k_custom', 4.0))])[0]
        else:
            nk_subst_fn = make_nk_fn([nk_subst_name])[0]

        wl_nm_ar   = np.linspace(wl_min, wl_max, wl_n, dtype=float)
        idex_0     = norder // 2
        order_list = list(range(-idex_0, idex_0 + 1))

        def _order_label(m): return '0 (specular)' if m == 0 else f'{m:+d}'
        def _idx_for_order(m): return idex_0 + m

        ocol1, ocol2 = st.columns(2)
        with ocol1:
            plot_orders = st.multiselect(
                'Orders to show in spectrum plot', order_list, default=[0],
                format_func=_order_label, key='plot_orders') or [0]
        with ocol2:
            repr_order = st.selectbox(
                'Order for colorimetry / CSV / 3D surface', order_list,
                index=order_list.index(0), format_func=_order_label, key='repr_order')

        _nonzero_idx = [_idx_for_order(m) for m in order_list if m != 0]

        def _diff_sum(ir):
            return np.clip(ir[:, _nonzero_idx].sum(axis=1), 0, 1) if _nonzero_idx else np.zeros(ir.shape[0])

        # ---- 計算コントロール（RCWA計算は「計算」ボタンを押した時のみ実行し、
        # 構造やパラメータのウィジェット操作だけでは自動再計算しない） ----
        with bar:
            _bcol1, _bcol2 = st.columns(2)
            with _bcol1:
                calc_clicked = st.button(
                    '▶ 計算', key='calc_btn', type='primary', width="stretch",
                    on_click=show_result_tab, args=('1d',),
                    help='現在の構造・パラメータでRCWA計算を実行します。')
                calc_clicked = calc_clicked or consume_autorun('1d')   # 「📚 解析記録」の「読み込んで再計算」
            with _bcol2:
                stop_clicked = st.button(
                    '⏹ 停止', key='stop_btn', width="stretch",
                    help='計算中にクリックすると、未着手の計算を取り消して打ち切ります。'
                         '直前に完了した計算結果があればそのまま表示され続けます。')
            _status_slot = st.empty()

        def _calc_signature(mode):
            base = (
                layers_cache_key(layers),
                pitch_um, norder, n_env,
                nk_subst_name, bool(st.session_state.get('custom_subst_nk', False)),
                st.session_state.get('subst_n_custom'), st.session_state.get('subst_k_custom'),
                wl_option, wl_min, wl_max, wl_n,
            )
            if mode == 'single':
                return ('single', base, inc_angle)
            return ('sweep', base, tuple(angle_range), angle_step, repr_order)

        def _run_calculation():
            """重いRCWA計算本体。「計算」ボタンが押された時のみ呼ばれる。
            掃引の全角度と色スウォッチ用の角度(0/30/60°)をまとめて1回で並列に解く。
            進捗バーの更新時に「停止」クリック等でStreamlitがスクリプトを中断すると、
            未着手の計算は取り消され、session_state.calc_cache は更新されない
            （=直前の結果がそのまま残る）。"""
            prog = st.progress(0.0, text='Calculating...')

            def _cb(done, total):
                prog.progress(done / total, text=f'S4 計算中 {done}/{total} 点')

            def _solve(angles):
                return calc_1d(wl_nm_ar, angles, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn,
                               progress_cb=_cb)

            if angle_mode == 'Single angle (2D)':
                irp, _, irs, _ = _solve([inc_angle])[float(inc_angle)]
                prog.empty()
                return {'mode': 'single', 'inc_angle': inc_angle, 'irp': irp, 'irs': irs}

            angle_ar = np.arange(angle_range[0], angle_range[1] + angle_step/2, angle_step, dtype=float)
            show_swatches = (wl_option == _SPECTRUM_MENU[0])
            res = _solve(list(angle_ar) + (_SWATCH_THETAS if show_swatches else []))
            prog.empty()

            idx_repr = _idx_for_order(repr_order)
            sweep = [res[float(a)] for a in angle_ar]
            swatch = {k: [] for k in ('Rp_single', 'Rs_single', 'Rp_diff', 'Rs_diff', 'Rp_total', 'Rs_total')}
            if show_swatches:
                for a in _SWATCH_THETAS:
                    s_irp, _, s_irs, _ = res[float(a)]
                    swatch['Rp_single'].append(s_irp[:, idx_repr].tolist())
                    swatch['Rs_single'].append(s_irs[:, idx_repr].tolist())
                    swatch['Rp_diff'].append(_diff_sum(s_irp).tolist())
                    swatch['Rs_diff'].append(_diff_sum(s_irs).tolist())
                    swatch['Rp_total'].append(np.clip(s_irp.sum(axis=1), 0, 1).tolist())
                    swatch['Rs_total'].append(np.clip(s_irs.sum(axis=1), 0, 1).tolist())

            return {
                'mode': 'sweep', 'angle_ar': angle_ar, 'repr_order': repr_order,
                'show_swatches': show_swatches,
                'Rp2d': np.array([r[0][:, idx_repr] for r in sweep]),
                'Rs2d': np.array([r[2][:, idx_repr] for r in sweep]),
                'Rp_diff_2d': np.array([_diff_sum(r[0]) for r in sweep]),
                'Rs_diff_2d': np.array([_diff_sum(r[2]) for r in sweep]),
                'Rp_total_2d': np.array([np.clip(r[0].sum(axis=1), 0, 1) for r in sweep]),
                'Rs_total_2d': np.array([np.clip(r[2].sum(axis=1), 0, 1) for r in sweep]),
                'swatch_thetas': _SWATCH_THETAS, 'swatch': swatch,
            }

        def _record_snapshot():
            """解析記録用に、計算した時点の条件を JSON にできる形で写し取る。"""
            custom = bool(st.session_state.get('custom_subst_nk', False))
            params = {
                'angle_mode': angle_mode, 'wl_option': wl_option, 'wl_min': wl_min, 'wl_max': wl_max, 'wl_n': wl_n,
                'n_env': n_env, 'substrate': nk_subst_name, 'custom_subst_nk': custom,
                'subst_n_custom': st.session_state.get('subst_n_custom') if custom else None,
                'subst_k_custom': st.session_state.get('subst_k_custom') if custom else None,
                'norder': norder, 'repr_order': repr_order, 'pitch_nm': pitch_nm, 'columns': columns,
                'graded_n_slices': st.session_state.get('graded_n_slices')
                if any(c.get('raise_mode') == 'graded' for c in columns) else None,
            }
            if angle_mode == 'Single angle (2D)':
                params['inc_angle'] = inc_angle
                theta = f'θ={inc_angle:g}°'
            else:
                params.update(angle_range=angle_range, angle_step=angle_step)
                theta = f'θ={angle_range[0]:g}〜{angle_range[1]:g}°（{angle_step:g}°刻み）'
            try:
                params = to_jsonable(params)
            except TypeError as e:
                params = {'_error': str(e)}
            subst = f'n={st.session_state.get("subst_n_custom")}+{st.session_state.get("subst_k_custom")}i' if custom else nk_subst_name
            text = (f'ピッチ {pitch_nm:g} nm, 列 {len(columns)} 本（{len(layers)} 層）, 基板 {subst}, 環境 n={n_env:g}, '
                    f'{theta}, λ={wl_min:g}〜{wl_max:g} nm（{wl_n} 点）, NumBasis {norder}')
            materials = list(_film_material_names(columns)) + ([] if custom else [nk_subst_name])
            return {'params': params, 'text': text, 'materials': [m for m in materials if isinstance(m, str)],
                    'default_title': f'1D ピッチ {pitch_nm:g} nm・列 {len(columns)} 本', 'norder': norder,
                    'repr_order': repr_order,
                    'fig_inputs': copy.deepcopy(dict(layers=layers, pitch_nm=pitch_nm, subst=nk_subst_name,
                                                     col_widths=col_widths))}

        current_sig = _calc_signature('single' if angle_mode == 'Single angle (2D)' else 'sweep')

        if calc_clicked:
            _t0 = time.time()
            with st.spinner('計算中...'):
                _result = _run_calculation()
            st.session_state.calc_cache = {
                'signature': current_sig,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'wl_nm_ar': wl_nm_ar,
                'result': _result,
                'record': _record_snapshot(),
                'elapsed_s': time.time() - _t0,
                'calc_id': uuid.uuid4().hex,
                'calc_time': _t0,
            }
        elif stop_clicked and st.session_state.get('calc_cache') is not None:
            st.info('⏹ 計算を停止しました。直前の計算結果を表示しています。')

        cache = st.session_state.get('calc_cache')
        if cache is None:
            _status_slot.caption('まだ計算していません')
            st.info('右下の「▶ 計算」を押すと、ここに結果が出ます。')
            st.stop()
        _status_slot.caption(f'最終計算: {cache["timestamp"]}')

        if cache['signature'] != current_sig:
            st.warning(f'⚠ 前回（{cache["timestamp"]}）の条件の結果です。変更を反映するには「▶ 計算」を押してください。')

        # 以降はキャッシュされた計算結果のみを参照する（ライブの wl_nm_ar もキャッシュ時点の
        # ものに揃え、stale時でも波長軸とスペクトル配列の長さが食い違わないようにする）。
        wl_nm_ar = cache['wl_nm_ar']
        result   = cache['result']

        if result['mode'] == 'single':
            irp, irs = result['irp'], result['irs']
            inc_angle = result['inc_angle']
            Rp = irp[:, _idx_for_order(repr_order)]
            Rs = irs[:, _idx_for_order(repr_order)]
            Rp_diff, Rs_diff = _diff_sum(irp), _diff_sum(irs)

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
            st.plotly_chart(fig, width="stretch")
            st.caption(f'色計算・CSV は m={_order_label(repr_order)} に基づきます。')

        else:
            angle_ar    = result['angle_ar']
            Rp2d,        Rs2d        = result['Rp2d'],        result['Rs2d']
            Rp_diff_2d,  Rs_diff_2d  = result['Rp_diff_2d'],  result['Rs_diff_2d']
            Rp_total_2d, Rs_total_2d = result['Rp_total_2d'], result['Rs_total_2d']
            swatch, show_swatches = result['swatch'], result['show_swatches']

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

            suffix = {_MODE_DIFF: 'diff', _MODE_TOTAL: 'total'}.get(disp_order_mode, 'single')
            Z_disp = {('Rp', 'diff'): Rp_diff_2d, ('Rs', 'diff'): Rs_diff_2d,
                      ('Rp', 'total'): Rp_total_2d, ('Rs', 'total'): Rs_total_2d,
                      ('Rp', 'single'): Rp2d, ('Rs', 'single'): Rs2d}[(disp_pol, suffix)]
            swatch_values = swatch[f'{disp_pol}_{suffix}'] if show_swatches else []

            n_filled = render_gsolver_style_display(
                lambdas=wl_nm_ar, thetas=angle_ar, values=Z_disp,
                show_swatches=show_swatches,
                swatch_thetas=result['swatch_thetas'], swatch_values=swatch_values,
            )
            if n_filled:
                st.warning(f'S4 が数値的に解けなかった点が {n_filled} 点あり、表示では同じ角度の'
                           '隣の波長から線形補間しています。')

            inc_angle = float(angle_ar[0])
            Rp, Rs = Rp2d[0, :], Rs2d[0, :]
            Rp_diff, Rs_diff = Rp_diff_2d[0, :], Rs_diff_2d[0, :]
            st.caption(f'色計算・CSV は代表角度 {round(inc_angle,1)}°, m={_repr_label} に基づきます。')

        # ================================================================
        # Colorimetry
        # ================================================================
        if wl_option == _SPECTRUM_MENU[0]:
            st.divider()
            if repr_order != 0:
                st.warning('高次回折光は強度が小さいため色が黒に近くなります。')
            swatches = [(f'Rp (m={_order_label(repr_order)})', Rp, 'cpRp'),
                        (f'Rs (m={_order_label(repr_order)})', Rs, 'cpRs'),
                        ('Rp 回折光 (m≠0)', Rp_diff, 'cpRpd'),
                        ('Rs 回折光 (m≠0)', Rs_diff, 'cpRsd')]
            for col, (label, spec, key) in zip(st.columns(4), swatches):
                XYZ, hex_ = _sd_to_hex(spec, wl_nm_ar, key)
                with col:
                    st.color_picker(label, hex_, key=key + hex_,
                                    help='回折光 = m≠0 の全次数の合計。ピッチ < 波長なら高次光が伝搬しないので黒。'
                                    if 'm≠0' in label else None)
                    st.write('XYZ', XYZ)

        # ================================================================
        # CSV Download
        # ================================================================
        d = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST')).strftime('%Y%m%d%H%M%S')
        st.divider()

        df_spec = pd.DataFrame(np.stack([Rp, Rs], axis=1), index=wl_nm_ar, columns=['Rp', 'Rs'])
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
            dfp = pd.DataFrame({c: _mir(Rp2d[:, i]) for i, c in enumerate(wl_cols)}, index=_xc)
            dfp.index.name = 'Angle(deg)'
            dfs = pd.DataFrame({c: _mir(Rs2d[:, i]) for i, c in enumerate(wl_cols)}, index=_xc)
            dfs.index.name = 'Angle(deg)'

            ac1, ac2 = st.columns(2)
            with ac1:
                st.download_button(f'Angle CSV Rp (m={_order_label(repr_order)})',
                                   convert_df(dfp), f'angle_Rp_{d}.csv', 'text/csv')
            with ac2:
                st.download_button(f'Angle CSV Rs (m={_order_label(repr_order)})',
                                   convert_df(dfs), f'angle_Rs_{d}.csv', 'text/csv')
            st.caption(f'行=角度(−{int(angle_ar[-1])}°～+{int(angle_ar[-1])}°)、列=各波長の反射率')

        # ================================================================
        # 解析記録（Supabase）
        # ================================================================
        snap = cache.get('record')
        if snap:
            render_save_panel('1d', {
                'params': snap['params'], 'materials': snap['materials'], 'default_title': snap['default_title'],
                'result': dict(result, wl_nm=cache['wl_nm_ar'], norder=snap['norder'],
                               repr_order=result.get('repr_order', snap['repr_order'])),
                'summary': {'text': snap['text'], 'angle_mode': result['mode']},
                'elapsed_s': cache['elapsed_s'], 'calc_id': cache['calc_id'], 'calc_time': cache['calc_time'],
                'figures_fn': lambda: _structure_figures(snap['fig_inputs']),
            })

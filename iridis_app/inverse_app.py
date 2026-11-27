"""「🎯 逆設計」のタブ: 目標の色（sRGB）または反射スペクトル（CSV）に近い構造を探す。

1D モードのタブ（平らな多層膜・1 次元の格子・1 次元の列構造）と、1D/2D に共通の部品（目標の入力、裏での探索と
停止、結果の表示）。2D モードのタブ（Mie 共鳴の粒子の配列）は inverse_mie_app.py。
探索は RCWA2D/inverse/（Streamlit 非依存）。基板・環境の屈折率はサイドバーの値を使う。探索は裏のスレッドで動かし、
途中で止められる。見つけた構造は「🧱 構造」に反映でき、ふだんの計算にかけられる。
"""
import copy
import io
import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from RCWA2D import color_metrics as cm
from RCWA2D.inverse import ColorTarget, SpectrumTarget, runner
from RCWA2D.inverse.families import film
from RCWA2D.inverse.families.columns import SteppedColumns
from RCWA2D.inverse.families.grating import Grating
from RCWA2D.inverse.targets import PRESET_COLORS
from RCWA2D.structure_builder import columns_to_layers

from .column_calc import calc_1d
from .common import build_nk_fn_map
from .log import get_logger

TAB_INVERSE = '🎯 逆設計'
_log = get_logger('iridis.inverse')

_KIND_LABEL = {'color': '色（sRGB）', 'spectrum': '反射スペクトル（CSV）'}
_POL_LABEL = {'avg': '偏光なし（s と p の平均）', 's': 's 偏光（Rs）', 'p': 'p 偏光（Rp）'}
_EFFORT_LABEL = {'quick': '速い', 'standard': 'ふつう', 'careful': '丁寧'}
_AGG_LABEL = {'mean': '角度の平均', 'max': '一番ずれた角度'}
_FAMILY_LABEL = {'film': '平らな多層膜', 'grating': '1 次元の格子（棒の並び）', 'columns': '1 次元の列構造（段差＋多層膜）'}
_COLOR_ANGLES = [(0.0, True, '#1f5fd0'), (30.0, False, '#1f5fd0'), (60.0, False, '#1f5fd0')]
_SHOW_CANDIDATES = 10
_INV_WIDTH_NM = 500.0   # エディタへ反映するときの列の幅（平らな膜なので結果には効かない。ピッチ < 波長で回折しない）
# runner.plan_wall の 1 あたりの秒数（実測 2026-11-26、8 ワーカー・速い探索: 円柱 0.057〜0.059、格子 0.049、列構造 0.036）
_SEC_PER_WALL = 0.055

# 入力欄の既定値。タブを離れると入力欄が描かれず Streamlit が値を捨てるので、控え（inv_saved）から戻す
_DEFAULTS = {'inv_kind': 'color', 'inv_ignore_l': False, 'inv_agg': 'mean', 'inv_spec_angle': 0.0, 'inv_family': 'film',
             'inv_mats': ['SiO2', 'TiO2'], 'inv_nmax': 8, 'inv_tmin': 20.0, 'inv_tmax': 300.0, 'inv_tstep': 1.0,
             'inv_total': 0.0, 'inv_pol': 'avg', 'inv_effort': 'standard', 'inv_tol': 0.0,
             'inv_gr_mats': ['TiO2'], 'inv_gr_p': (160, 400), 'inv_gr_h': (40, 300), 'inv_gr_wmin': 40.0, 'inv_gr_gap': 40.0,
             'inv_gr_nodiff': True,
             'inv_co_mats': ['SiO2', 'TiO2'], 'inv_co_w': (100, 400), 'inv_co_h': (20, 250), 'inv_co_t': (30, 150),
             'inv_co_n': (3, 5),
             'inv_mie_mats': ['cSi_Green2008'], 'inv_mie_shape': 'disk', 'inv_mie_lattice': 'square', 'inv_mie_p': (100, 400),
             'inv_mie_d': (50, 300), 'inv_mie_h': (40, 300), 'inv_mie_gap': 20.0, 'inv_mie_aspect': 2.0,
             'inv_mie_side': 'none', 'inv_mie_wl': 380.0}
for _th, _on, _hex in _COLOR_ANGLES:
    _DEFAULTS.update({f'inv_use_{int(_th)}': _on, f'inv_hex_{int(_th)}': _hex})


def restore_inputs(nk_namelist):
    """入力欄を作る前に、値が無いものだけ控え（無ければ既定値）を入れる。"""
    ss = st.session_state
    saved = ss.get('inv_saved', {})
    for k, v in _DEFAULTS.items():
        if k not in ss:
            v = saved.get(k, v)
            ss[k] = [m for m in v if m in nk_namelist] if k.endswith('mats') else v


def remember_inputs():
    ss = st.session_state
    ss['inv_saved'] = {**ss.get('inv_saved', {}), **{k: ss[k] for k in _DEFAULTS if k in ss}}


# ---------------------------------------------------------------- 目標
def parse_spectrum_csv(data):
    """CSV（1 列目 = 波長、2 列目 = 反射率）→ (波長 nm, 反射率 0〜1)。
    反射率の最大が 1.5 を超えれば % とみなして 100 で割り、波長の最大が 5 未満なら µm とみなして 1000 倍する。
    数値でない行（見出しなど）は飛ばす。"""
    df = pd.read_csv(io.BytesIO(data), header=None, sep=None, engine='python')
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='any', subset=[0, 1])
    if len(df) < 3:
        raise ValueError('数値の行が 3 行未満です（1 列目 = 波長、2 列目 = 反射率）。')
    wl, R = df[0].to_numpy(float), df[1].to_numpy(float)
    if wl.max() < 5:
        wl = wl * 1000.0
    if R.max() > 1.5:
        R = R / 100.0
    order = np.argsort(wl)
    return wl[order], np.clip(R[order], 0.0, 1.0)


def _set_preset(hex_):
    """青・緑・赤のボタン: チェックの入った角度すべてに同じ色を入れる（どれも入っていなければ 0° に）。"""
    ss = st.session_state
    angles = [int(th) for th, _, _ in _COLOR_ANGLES]
    on = [a for a in angles if ss.get(f'inv_use_{a}')]
    if not on:
        on = [0]
        ss['inv_use_0'] = True
    for a in on:
        ss[f'inv_hex_{a}'] = hex_


def _add_outlined_line(fig, x, y, color, name, dash=None):
    """色の線（白や黄色でも見えるよう、下に灰色の太い線を敷く）。"""
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='rgba(120,120,120,.55)', width=5, dash=dash),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color, width=2.5, dash=dash),
                             hovertemplate='%{x:.0f} nm: %{y:.3f}<extra></extra>'))


def _preview_layout(fig):
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', y=1.15),
                      xaxis_title='波長 [nm]', yaxis_title='反射率', yaxis_range=[0, 1.02], xaxis_range=[380, 780])
    return fig


def _target_preview(angles, hexes):
    """目標の色ごとに、その色になる反射スペクトルの例を描く（逆設計の前に、目標が反射率でどうなるかを見る）。"""
    groups = {}
    for a, h in zip(angles, hexes):
        groups.setdefault(h.lower(), []).append(a)
    fig = go.Figure()
    bad = []
    for h, angs in groups.items():
        R, got, de = cm.example_spectrum(h)
        _add_outlined_line(fig, cm.CIE_WL, R, h, '・'.join(f'{a:g}°' for a in angs) + f' {h}')
        if de > 1.0:
            bad.append(f'{h}（出せる一番近い色 {got}、ΔE₀₀ {de:.1f}）')
    st.plotly_chart(_preview_layout(fig), key='inv_target_fig')
    st.caption('目標の色になる反射スペクトルの例。同じ色になるスペクトルは無数にあり、これはその 1 つ'
               '（なめらかで、反射率がなるべく小さいもの）。山の位置と高さ（明るさ）の目安に。',
               help='反射率を 0〜1 に限って、目標の色（D65 光源・2° 視野）とぴったり同じ色になるスペクトルのうち、'
                    '凹凸が少なく全体に反射率の小さいものを選んでいる。実際の構造のスペクトルは、山の幅や形がこれと違っても同じ色になりうる。')
    if bad:
        st.warning('反射率 ≤ 1 では出せない色です: ' + '、'.join(bad) + '。「明るさは問わない」にすると色みだけ合わせられます。')


def render_target():
    """目標の入力欄 → (ColorTarget / SpectrumTarget, 説明, エラー文)。"""
    kind = st.radio('目標', list(_KIND_LABEL), format_func=_KIND_LABEL.get, horizontal=True, key='inv_kind',
                    help='色: 入射角ごとに目標の色を決める（1 つでもよい）。比べるのは CIELAB 上の色差 ΔE₀₀ '
                         '（D65 光源・2° 視野。「📈 計算結果」の色と同じ計算）。'
                         'ΔE₀₀ の目安: 1 未満 = 並べても差がほぼ分からない、2〜3 = 並べると分かる、5 以上 = はっきり違う。')
    if kind == 'color':
        box = st.container(horizontal=True, vertical_alignment='center')
        for name, hx in PRESET_COLORS.items():
            box.button(f'{ {"青": "🟦", "緑": "🟩", "赤": "🟥"}[name] } {name}', key=f'inv_preset_{name}', on_click=_set_preset, args=(hx,),
                       help=f'チェックの入った角度すべてを {hx} にする。そのあと下の欄で角度ごとに直せる。')
        angles, hexes = [], []
        for col, (theta, _, _) in zip(st.columns(3), _COLOR_ANGLES):
            with col:
                use = st.checkbox(f'{theta:g}° の色', key=f'inv_use_{int(theta)}')
                hx = st.color_picker(f'{theta:g}° の目標', key=f'inv_hex_{int(theta)}', disabled=not use,
                                     label_visibility='collapsed')
                if use:
                    angles.append(theta)
                    hexes.append(hx)
        o1, o2 = st.columns(2)
        with o1:
            ignore_l = st.checkbox('明るさは問わない（色相と彩度だけ合わせる）', key='inv_ignore_l',
                                   help='構造色は鮮やかでも暗くなりやすく、明るく鮮やかな色は反射率 ≤ 1 では出せないことがある。'
                                        'そのときは明るさを外すと色みの合う構造が見つかりやすい。')
        with o2:
            agg = st.radio('角度が複数のとき', list(_AGG_LABEL), format_func=_AGG_LABEL.get, horizontal=True,
                           key='inv_agg', disabled=len(angles) < 2,
                           help='角度の平均: 全体として近い構造。一番ずれた角度: どの角度でもずれが小さい構造。')
        if not angles:
            return None, '', '目標の色を 1 つ以上選んでください。'
        _target_preview(angles, hexes)
        desc = '・'.join(f'{a:g}° {h}' for a, h in zip(angles, hexes)) + ('（明るさは問わない）' if ignore_l else '')
        return ColorTarget(angles, hexes, ignore_lightness=ignore_l, aggregate=agg), desc, None

    s1, s2 = st.columns([3, 1])
    with s1:
        up = st.file_uploader('反射スペクトルの CSV（1 列目 = 波長 nm、2 列目 = 反射率 0〜1 または %）', type=['csv', 'txt'],
                              key='inv_csv')
    with s2:
        angle = st.number_input('入射角 θ [deg]', 0.0, 89.0, step=1.0, key='inv_spec_angle')
    if up is None:
        return None, '', 'CSV を選んでください。'
    try:
        wl, R = parse_spectrum_csv(up.getvalue())
    except Exception as e:   # 形の違う CSV など
        return None, '', f'CSV を読めません: {e}'
    st.caption(f'{up.name}: {wl[0]:g}〜{wl[-1]:g} nm、{wl.size} 点（反射率 {R.min():.2f}〜{R.max():.2f}）。'
               'ずれは反射率の差の二乗平均（RMSE、% 単位）で測る。')
    hx = cm.xyz_to_hex(cm.spectrum_to_xyz(cm.to_cie_grid(wl, R)))
    fig = go.Figure()
    _add_outlined_line(fig, wl, R, hx, f'目標（この反射率の色 {hx}）')
    st.plotly_chart(_preview_layout(fig), key='inv_target_fig')
    return SpectrumTarget(wl, R, angle), f'{up.name}（θ={angle:g}°）', None


# ---------------------------------------------------------------- 設定
def render_common_settings(show_tol=True):
    e1, e2, e3 = st.columns(3)
    with e1:
        pol = st.radio('偏光', list(_POL_LABEL), format_func=_POL_LABEL.get, key='inv_pol',
                       help='ふつうの照明（偏光していない光）なら s と p の平均。')
    with e2:
        effort = st.radio('探索の細かさ', list(_EFFORT_LABEL), format_func=_EFFORT_LABEL.get, horizontal=True,
                          key='inv_effort', help='丁寧なほど見落としが減るが時間がかかる。')
    with e3:
        tol = st.number_input('ここまで近づいたら止める', 0.0, 50.0, step=0.5, key='inv_tol',
                              help='ΔE₀₀（色）または RMSE %（スペクトル）。0 なら最後まで探す。')
    return pol, effort, float(tol)


def step_input(label='寸法の単位 [nm]'):
    return st.number_input(label, 0.1, 100.0, step=1.0, key='inv_tstep', format='%g',
                           help='見つける寸法をこの値の整数倍にする（1 なら 62 nm、63 nm…。0.1 なら 0.1 nm 単位、5 なら 5 nm 単位）。'
                                '粗いほど作りやすいが、目標からのずれは大きくなりやすい。')


def time_caption(families, target, settings):
    if not families or target is None:
        return
    plans = [runner.plan_wall(f, target, settings) for f in families]
    sec = sum(p[0] for p in plans) * _SEC_PER_WALL
    info = plans[0][1]
    conc = min(p[1]['concurrency'] for p in plans)
    t = f'約 {sec:.0f} 秒' if sec < 90 else f'約 {sec / 60:.0f} 分'
    par = (f'{info["workers"]} 並列' if conc >= info['workers']
           else f'空きメモリ {info["mem_gb"]:.1f} GB のため、細かい計算は {conc} 並列（ワーカー {info["workers"]}）')
    st.caption(f'材料の組み合わせ {len(families)} 通り・{len(target.angles)} 角度・{par}・所要時間の目安 {t}',
               help='粗い次数（NumBasis）で広く探し、細かい次数で寸法を直してから、さらに細かい次数で確かめる。'
                    '目安は、ワーカーの数（環境変数 IRIDIS_WORKERS、既定は CPU の数）と、空きメモリに収まる同時実行数から'
                    '計算する。ほかの計算が動いているときなど、PC の状態によって変わる。')


def _film_settings(nk_namelist):
    mats = st.multiselect('使ってよい材料', nk_namelist, key='inv_mats',
                          help='1 つの膜に使うのは 2 種類まで（交互に積む）。3 つ以上選ぶと、2 つずつの組み合わせを全部探す。')
    pairs = film.material_pairs(mats)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_max = st.number_input('層の数の上限', 1, 30, step=1, key='inv_nmax',
                                help='上から A, B, A, … と交互に積む。薄すぎる層は無くなるので、実際の層の数はこれ以下になる。')
    with c2:
        t_min = st.number_input('最小膜厚 [nm]', 0.0, 500.0, step=5.0, key='inv_tmin',
                                help='これより薄い層は作らない（探索の中で 0 nm、つまり層が無いものとして扱う）。')
    with c3:
        t_max = st.number_input('1 層の最大膜厚 [nm]', 10.0, 5000.0, step=10.0, key='inv_tmax')
    with c4:
        t_step = step_input('膜厚の単位 [nm]')
    with c5:
        total = st.number_input('総膜厚の上限 [nm]', 0.0, 1e5, step=100.0, key='inv_total', help='0 なら制限なし。')
    pol, effort, tol = render_common_settings()
    if t_min >= t_max:
        return mats, None, '最小膜厚は 1 層の最大膜厚より小さくしてください。'
    if film.max_on_grid(t_max, t_step) < max(t_min, t_step):
        return mats, None, (f'最小膜厚 {t_min:g} nm と最大膜厚 {t_max:g} nm の間に、{t_step:g} nm の整数倍がありません。'
                            '膜厚の単位を小さくするか、最大膜厚を大きくしてください。')
    if not mats:
        return mats, None, '材料を 1 つ以上選んでください。'
    runs = len(pairs) * film.EFFORTS[effort]['restarts']
    st.caption(f'材料の組み合わせ {len(pairs)} 通り（{"、".join("・".join(p) for p in pairs[:6])}'
               f'{" ほか" if len(pairs) > 6 else ""}）・探索 {runs} 回。1 回あたり数秒〜数十秒（層の数・角度の数・細かさによる。'
               '例: 8 層・1 角度・ふつうで約 7 秒、3 角度で約 20 秒）。')
    return mats, film.SearchSettings(n_max_layers=int(n_max), t_min_nm=float(t_min), t_max_nm=float(t_max), t_step_nm=float(t_step),
                                     total_max_nm=float(total), pol=pol, effort=effort, tol=tol), None


def _grating_settings(nk_namelist, target, n_env, nk_subst_fn):
    mats = st.multiselect('棒の材料', nk_namelist, key='inv_gr_mats', help='複数選ぶと 1 つずつ全部探す。')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p = st.slider('周期 [nm]', 80, 1000, step=10, key='inv_gr_p')
        nodiff = st.checkbox('回折が出ない周期に限る', key='inv_gr_nodiff',
                             help='空気側に鏡面反射（0 次）だけが出る周期 < 380 nm ÷（環境の n ＋ sinθ）に限る。基板側に出る回折は許す。'
                                  '外すと上の範囲をそのまま使う（その場合、色は 0 次の反射だけで計算する）。')
    with c2:
        h = st.slider('高さ [nm]', 10, 1000, step=10, key='inv_gr_h')
    with c3:
        w_min = st.number_input('棒の幅の最小 [nm]', 5.0, 500.0, step=5.0, key='inv_gr_wmin')
        gap = st.number_input('すき間の最小 [nm]', 5.0, 500.0, step=5.0, key='inv_gr_gap')
    with c4:
        step = step_input()
    pol, effort, tol = render_common_settings()
    if not mats:
        return mats, None, None, '材料を 1 つ以上選んでください。'
    if target is None:
        return mats, None, None, None
    nd = dict(angles_deg=target.angles, n_env=n_env, nk_sub_fn=nk_subst_fn, side='env') if nodiff else None
    try:
        fams = [Grating(m, p, h, w_min, gap, step, nd) for m in mats]
    except ValueError as e:
        return mats, None, None, (f'{e}' + ('「回折が出ない周期に限る」を外すか、角度や周期の範囲を変えてください。'
                                           if nodiff else ''))
    if nd:
        st.caption(f'探す周期の上限: {fams[0].variables[0].hi:.0f} nm（回折が出ない範囲）')
    s = runner.Settings(pol=pol, effort=effort, tol=tol)
    time_caption(fams, target, s)
    return mats, fams, s, None


def _columns_settings(nk_namelist, target):
    mats = st.multiselect('膜に使ってよい材料（2 つ以上）', nk_namelist, key='inv_co_mats',
                          help='1 つの構造に使うのは 2 種類。3 つ以上選ぶと 2 つずつの組み合わせを全部探す。段差は基板の材質。')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w = st.slider('列の幅 [nm]', 50, 1500, step=10, key='inv_co_w', help='段差の上の列・下の列のそれぞれの幅。合計が周期になる。')
        step = step_input()
    with c2:
        h = st.slider('段差の高さ [nm]', 0, 600, step=5, key='inv_co_h')
    with c3:
        t = st.slider('1 層の膜厚 [nm]', 5, 400, step=5, key='inv_co_t')
    with c4:
        n = st.slider('ペアの数', 1, 15, step=1, key='inv_co_n', help='層の数が増えると計算時間も増える。')
    pol, effort, tol = render_common_settings()
    if len(mats) < 2:
        return mats, None, None, '材料を 2 つ以上選んでください。'
    if target is None:
        return mats, None, None, None
    fams = [SteppedColumns(pr, w, h, t, n, step) for pr in film.material_pairs(mats)]
    s = runner.Settings(pol=pol, effort=effort, tol=tol)
    time_caption(fams, target, s)
    return mats, fams, s, None


# ---------------------------------------------------------------- 裏での探索
def job_running(key):
    job = st.session_state.get(key)
    return job is not None and job['thread'].is_alive()


def start_job(key, meta, fn):
    """fn(progress_cb, stop_event) を裏のスレッドで動かす。スレッドは st.session_state に触れず、渡した dict だけを書く。"""
    job = {'stop': threading.Event(), 'info': {}, 't0': time.time(), 'error': None, 'out': None, 'meta': meta}

    def work():
        _log.info('逆設計 %s: 開始 %s', key, meta.get('desc', ''))
        try:
            job['out'] = fn(lambda info: job.update(info=info), job['stop'])
            _log.info('逆設計 %s: %s（%.0f 秒、最良 %s）', key, '停止' if job['stop'].is_set() else '完了', time.time() - job['t0'],
                      f"{job['out'][0]['loss']:.2f}" if job['out'] else 'なし')
        except Exception as e:   # スレッドの例外は自動では伝わらないので、画面に出すために控える
            _log.exception('逆設計 %s: エラー', key)
            job['error'] = f'{type(e).__name__}: {e}'
        job['elapsed_s'] = time.time() - job['t0']

    job['thread'] = threading.Thread(target=work, daemon=True)
    st.session_state[key] = job
    job['thread'].start()


def collect_job(key, result_key):
    """終わった探索の結果を session_state に移す。"""
    job = st.session_state.get(key)
    if job is None or job['thread'].is_alive():
        return
    del st.session_state[key]
    if job['error']:
        st.error(f"探索中にエラーが発生しました: {job['error']}")
        return
    st.session_state[result_key] = {**job['meta'], 'cands': job['out'] or [], 'elapsed_s': job.get('elapsed_s', 0.0),
                                    'finished_at': time.strftime('%H:%M:%S'), 'stopped': job['stop'].is_set()}
    for k in ('inv_s4_rows', result_key + '_pick'):
        st.session_state.pop(k, None)


def _stop_job(key):
    job = st.session_state.get(key)
    if job is not None:
        job['stop'].set()


@st.fragment(run_every=1.0)
def render_progress(key):
    job = st.session_state.get(key)
    if job is None or not job['thread'].is_alive():
        st.rerun()
    info, el = job['info'], time.time() - job['t0']
    frac = float(info.get('frac', 0.0))
    unit = job['meta'].get('unit', '')
    parts = [info.get('text') or info.get('stage', '準備中')]
    if info.get('n_families', 1) > 1:
        parts.insert(0, f"{info['family'] + 1}/{info['n_families']}（{info['label']}）")
    if info.get('best') is not None:
        parts.append(f"いまの最良 {unit} {info['best']:.2f}")
    parts.append(f'経過 {el:.0f} 秒' + (f'・残り約 {el / frac * (1 - frac):.0f} 秒' if frac > 0.03 else ''))
    c1, c2 = st.columns([5, 1], vertical_alignment='center')
    c1.progress(frac, text='・'.join(parts))
    c2.button('⏹ 停止', key=key + '_stop', on_click=_stop_job, args=(key,), disabled=job['stop'].is_set(), width='stretch')
    if job['stop'].is_set():
        st.caption('停止を受け付けました。実行中の計算が終わり次第、そこまでの候補を出します。')


def target_meta(target, desc):
    tgt = ({'hexes': list(target.hexes), 'angles_deg': list(target.angles_deg), 'ignore_lightness': target.ignore_lightness}
           if target.kind == 'color' else {'wl_nm': target.wl_nm.tolist(), 'R': target.R.tolist(), 'angle_deg': target.angle_deg})
    return {'kind': target.kind, 'target': tgt, 'desc': desc, 'unit': 'ΔE₀₀' if target.kind == 'color' else 'RMSE %'}


# ---------------------------------------------------------------- 構造へ反映
def _apply_to_editor(cand):
    """見つけた構造を編集画面へ書き込む（on_click なので入力欄が作られる前に書ける）。id は新しく振る。"""
    ss = st.session_state
    if cand['family'] == 'film':
        # 列は 1 本。エディタの膜は下から積むので、上から並んだ layers を逆にする
        cols = [{'width': _INV_WIDTH_NM, 'subst_raise': 0.0, 'raise_mode': 'sharp',
                 'film_layers': [{'type': 'simple', 'material': m, 'thickness': float(t)} for m, t in reversed(cand['layers'])]}]
        msg = f'列 1 本・幅 {_INV_WIDTH_NM:g} nm・{len(cand["layers"])} 層'
    else:
        ap = cand['apply']
        msg = cand['summary']
        if ap['mode'] == '2d':
            n = int(ss.get('_array_layer_id', 0) or 0)
            layers = copy.deepcopy(ap['array_layers'])
            for lyr in layers:
                n += 1
                lyr['id'] = n
            ss.update(array_layers=layers, _array_layer_id=n, pitch_x=float(ap['pitch_nm']),
                      lattice_shape='六方格子' if ap['lattice'] == 'hex' else '正方格子',
                      lens_enabled=False, step_enabled=False, sc_enabled=False)
            ss['inv_flash'] = (f'「🧱 構造」に反映しました（{msg}）。それまでの層構成は置き換わり、基板段差・レンズ状の凹凸・'
                               f'周期領域は使わない設定になっています。ふだんの計算では NumBasis を {cand["norder"]} にすると同じ値になります。')
            return
        cols = copy.deepcopy(ap['columns'])
    n_col, n_lay = int(ss.get('_ncol_id', 0) or 0), int(ss.get('_nlayer_id', 0) or 0)
    for col in cols:
        n_col += 1
        col['id'] = n_col
        for fl in col['film_layers']:
            n_lay += 1
            fl['id'] = n_lay
    ss.update(columns=cols, _ncol_id=n_col, _nlayer_id=n_lay)
    ss['inv_flash'] = f'「🧱 構造」に反映しました（{msg}）。それまでの列は置き換わっています。'


def _swatch(hex_, label):
    """白枠で囲んだ色の四角と、その下のラベル（HTML）。"""
    return ('<span style="display:inline-flex;flex-direction:column;align-items:center;margin-right:6px">'
            f'<span style="display:inline-block;width:28px;height:28px;background:{hex_};border:3px solid #fff;'
            'border-radius:4px;box-shadow:0 0 0 1px rgba(128,128,128,.55)"></span>'
            f'<span style="font-size:11px;opacity:.75">{label}</span></span>')


def _swatch_html(angles, targets, founds):
    """左に目標の色、右に見つけた構造の色（角度ごと）を並べる（HTML）。目標が全角度で同じなら 1 つにまとめる。"""
    left = ''
    if targets and any(targets):
        if len({t.lower() for t in targets}) == 1:
            left = _swatch(targets[0], '目標')
        else:
            left = ''.join(_swatch(t, f'目標 {a:g}°') for a, t in zip(angles, targets))
        left += '<span style="font-size:16px;opacity:.6;margin:0 6px 14px 0">→</span>'
    right = ''.join(_swatch(f, f'{a:g}°') for a, f in zip(angles, founds))
    return f'<span style="display:inline-flex;align-items:center;flex-wrap:wrap">{left}{right}</span>'


def _layers_text(layers):
    return ' / '.join(f'{m} {t:.7g}' for m, t in layers) if layers else '（膜なし）'


def _s4_check(cand, res, nk_subst_fn, n_env):
    """候補を S4 で計算し、伝達行列法と同じ値になるかを返す。"""
    layers = cand['layers']
    col = {'id': 0, 'width': _INV_WIDTH_NM, 'subst_raise': 0.0, 'raise_mode': 'sharp',
           'film_layers': [{'id': i, 'type': 'simple', 'material': m, 'thickness': t}
                           for i, (m, t) in enumerate(reversed(layers))]}
    wl = np.asarray(cand['wl_nm'], float)
    out = calc_1d(wl, cand['angles_deg'], _INV_WIDTH_NM / 1000.0, 3, columns_to_layers([col]),
                  build_nk_fn_map([m for m, _ in layers]), n_env, nk_subst_fn)
    pol = res['settings']['pol']
    rows = []
    for i, th in enumerate(cand['angles_deg']):
        irp, _, irs, _ = out[float(th)]
        R = {'s': irs[:, 1], 'p': irp[:, 1]}.get(pol, 0.5 * (irs[:, 1] + irp[:, 1]))
        R_tmm = np.asarray(cand['R'][i])
        row = {'θ [deg]': th, 'S4 と伝達行列法の差（最大）': float(np.max(np.abs(R - R_tmm)))}
        if res['kind'] == 'color':
            lab = cm.spectrum_to_lab(cm.to_cie_grid(wl, R))
            row['S4 の色'] = cm.xyz_to_hex(cm.spectrum_to_xyz(cm.to_cie_grid(wl, R)))
            row['ΔE₀₀（S4）'] = float(cm.delta_e2000(lab, cm.hex_to_lab(res['target']['hexes'][i]),
                                                     ignore_lightness=res['target']['ignore_lightness']))
        else:
            row['RMSE %（S4）'] = float(100 * np.sqrt(np.mean((R - np.interp(wl, res['target']['wl_nm'],
                                                                         res['target']['R'])) ** 2)))
        rows.append(row)
    return rows


def _detail_fig(cand, res):
    fig = go.Figure()
    for i, th in enumerate(cand['angles_deg']):
        fig.add_trace(go.Scatter(x=cand['wl_nm'], y=cand['R'][i], mode='lines', name=f'見つけた膜 θ={th:g}°',
                                 line=dict(color=cand['hexes'][i] if res['kind'] == 'color' else None, width=2.5)))
    if res['kind'] == 'spectrum':
        fig.add_trace(go.Scatter(x=res['target']['wl_nm'], y=res['target']['R'], mode='markers', name='目標',
                                 marker=dict(size=5, color='#555')))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', y=1.12),
                      xaxis_title='波長 [nm]', yaxis_title='反射率', yaxis_range=[0, 1])
    return fig


def _period_text(pc):
    """周期を ±5% 動かしたときのピークの動きの説明。"""
    if not pc or not pc.get('peak_nm') or pc.get('ratio') is None:
        return None
    moved = '、'.join(f'{dp:+g}% → {q:.0f} nm' for dp, q in pc['moved'] if q)
    r = pc['ratio']
    verdict = ('周期にほとんど依存しない（Mie 共鳴が主）' if r < 0.3 else
               '周期の影響もある（Mie 共鳴と格子の両方）' if r < 0.7 else '周期と一緒に動く（格子の影響が大きい）')
    return f'反射のピーク {pc["peak_nm"]:.0f} nm。粒子はそのままで周期だけ動かすと {moved}。{verdict}。'


def render_results(res, result_key, nk_subst_fn, n_env):
    cands = res['cands']
    unit = res['unit']
    st.caption(f'{res["family_label"]}｜目標 {res["desc"]}・基板 {res["subst"]}・環境 n={res["n_env"]:g}・{_POL_LABEL[res["settings"]["pol"]]}・'
               f'{res["step_nm"]:g} nm 単位・{res["elapsed_s"]:.0f} 秒（{res["finished_at"]}）' + ('・途中で停止' if res.get('stopped') else ''))
    if not cands:
        st.warning('候補がありません（すぐに停止したなど）。')
        return
    best = cands[0]['loss']
    if res['kind'] == 'color' and best > 5:
        st.warning(f'一番近い構造でも ΔE₀₀ = {best:.1f} です。明るく鮮やかな色は、反射率 ≤ 1 の制約で出せないことが'
                   'あります。「明るさは問わない」にする、材料や寸法の範囲を広げる、角度を減らす、を試してください。')
    if any(not c.get('verified', True) for c in cands):
        st.warning('途中で止めたので、粗い計算のままの値です。実際の色とは違うことがあります。')
    for k, c in enumerate(cands[:_SHOW_CANDIDATES]):
        c1, c2, c3, c4 = st.columns([0.5, 2.8, 1.0, 3.7], vertical_alignment='center')
        c1.markdown(f'**#{k + 1}**')
        tg = res['target']['hexes'] if res['kind'] == 'color' else None
        c2.markdown(_swatch_html(c['angles_deg'], tg, c['hexes']), unsafe_allow_html=True)
        c3.markdown(f'{unit} **{c["loss"]:.2f}**')
        if c['family'] == 'film':
            text = (f'{"・".join(c["pair"])}｜{c["n_layers"]} 層・総 {c["total_nm"]:.7g} nm  \n'
                    f'<span style="font-size:12px;opacity:.8">上から {_layers_text(c["layers"])}</span>')
        else:
            text = (f'{c["summary"]}  \n<span style="font-size:12px;opacity:.8">NumBasis {c["norder"]} の値'
                    + (f'（{c["norder_coarse"]} では {c["loss_coarse"]:.2f}）' if 'loss_coarse' in c else '') + '</span>')
        c4.markdown(text, unsafe_allow_html=True)
    if res['kind'] == 'color':
        st.caption('色の四角は、左が目標、→ の右が見つけた構造の色（角度ごと）。同じ色になる構造は何通りもあるので、'
                   '作りやすいものを選ぶとよい。')

    st.divider()
    n_show = min(len(cands), _SHOW_CANDIDATES)
    pick = st.selectbox('詳しく見る候補', range(n_show), format_func=lambda i: f'#{i + 1}（{unit} {cands[i]["loss"]:.2f}）',
                        key=result_key + '_pick')
    cand = cands[min(pick, n_show - 1)]
    d1, d2 = st.columns([1, 2])
    with d1:
        if cand['family'] == 'film':
            st.dataframe(pd.DataFrame({'材料': [m for m, _ in cand['layers']],
                                       '厚さ [nm]': [float(t) for _, t in cand['layers']]},
                                      index=pd.RangeIndex(1, len(cand['layers']) + 1, name='上から')), width='stretch')
        else:
            st.dataframe(pd.DataFrame(cand['dims'], columns=['寸法', '値 [nm]']), hide_index=True, width='stretch')
        st.button('この構造を「🧱 構造」に反映', key=result_key + '_apply', type='primary', on_click=_apply_to_editor, args=(cand,),
                  help='いま編集中の構造を、見つけた構造に置き換える。そのあとふだんの計算・表示ができる。')
        if cand['family'] == 'film' and st.button('S4 で確かめる', key='inv_s4'):
            with st.spinner('S4 で計算中…'):
                st.session_state['inv_s4_rows'] = (id(cand), _s4_check(cand, res, nk_subst_fn, n_env))
    with d2:
        st.plotly_chart(_detail_fig(cand, res), width='stretch')
    if cand.get('mie_sphere'):
        t = _period_text(cand.get('period_check'))
        if t:
            st.info(t)
        ms = cand['mie_sphere']
        f = lambda v: f'{v:.0f} nm' if v else '範囲外'
        st.caption(f'参考（Mie 理論）: 同じ体積の孤立した球（直径 {ms["diameter_nm"]:.0f} nm）の共鳴は、磁気双極子 {f(ms["magnetic"])}・'
                   f'電気双極子 {f(ms["electric"])}。', help='まわりは環境と同じ媒質。配列では基板・隣の粒子・形の違いで波長がずれる。')
    dif = cand.get('diffraction')
    if dif:
        parts = '、'.join(f'{a:g}° で {100 * m:.0f}%（{w:.0f} nm）' if m >= 0.005 else f'{a:g}° でなし'
                         for a, m, w in dif['per_angle'])
        msg = f'空気側に出る回折光（鏡面以外の向き）の最大: {parts}。'
        (st.warning if dif['max'] >= 0.1 else st.caption)(
            msg + ('見る向きによって色が変わって見えることがある。' if dif['max'] >= 0.1 else ''),
            **({} if dif['max'] >= 0.1 else {'help': '入射光に対する割合。色の評価は鏡面反射（R0）だけで、回折光は入れていない。'}))
    rows = st.session_state.get('inv_s4_rows')
    if rows and rows[0] == id(cand):
        st.dataframe(pd.DataFrame(rows[1]), hide_index=True, width='stretch')
        st.caption('平らな膜なので、伝達行列法と S4 は丸め誤差の範囲で一致するはず（差が 1e-6 未満）。')


def render_run(job_key, result_key, ready, start, nk_subst_fn, n_env, blocked=None):
    """「探索を始める」・進み具合・結果。blocked: 始められない理由（あれば）。"""
    running = job_running(job_key)
    if st.button('🎯 探索を始める', key='inv_run', type='primary', disabled=running or not ready or bool(blocked), help=blocked):
        start()
        st.rerun()
    if running:
        render_progress(job_key)
    res = st.session_state.get(result_key)
    if res and not running:
        st.divider()
        render_results(res, result_key, nk_subst_fn, n_env)


def render_inverse_tab(nk_namelist, n_env, subst_label, nk_subst_fn):
    """1D モードの逆設計タブ。"""
    flash = st.session_state.pop('inv_flash', None)
    if flash:
        st.success(flash)
    collect_job('inv_job_1d', 'inv1d')
    restore_inputs(nk_namelist)
    family = st.radio('探す構造', list(_FAMILY_LABEL), format_func=_FAMILY_LABEL.get, horizontal=True, key='inv_family',
                      help='多層膜: 2 種類の材料を交互に積んだ平らな膜。膜厚は層ごとに別。計算は伝達行列法（平らな膜では S4 と同じ答え）。\n\n'
                           '格子: 基板の上に、断面が長方形の棒を周期的に並べた構造。1 種類の材料。計算は S4。\n\n'
                           '列構造: 基板に段差をつけ、2 種類の材料の多層膜を段差に沿って積んだ構造。周期は段差の上と下の 2 列。計算は S4。\n\n'
                           '基板と環境の屈折率はサイドバーの設定を使う。Mie 共鳴（円柱・球の配列）は「2次元周期アレイ (2D)」の逆設計タブ。')
    target, desc, err = render_target()
    st.divider()
    fams = None
    if family == 'film':
        mats, settings, err2 = _film_settings(nk_namelist)
    elif family == 'grating':
        mats, fams, settings, err2 = _grating_settings(nk_namelist, target, n_env, nk_subst_fn)
    else:
        mats, fams, settings, err2 = _columns_settings(nk_namelist, target)
    remember_inputs()
    if err:
        st.caption(err)
    if err2:
        st.error(err2)

    def start():
        nk = build_nk_fn_map(mats)
        meta = {**target_meta(target, desc), 'subst': subst_label, 'n_env': float(np.real(n_env)), 'materials': list(mats),
                'settings': vars(settings).copy(), 'step_nm': float(st.session_state['inv_tstep']), 'family': family,
                'family_label': _FAMILY_LABEL[family]}
        if family == 'film':
            def fn(cb, stop):
                def film_cb(i):
                    cb({'frac': min((i['run'] + min(i['gen'] / i['maxiter'], 1.0)) / i['n_runs'], 1.0), 'best': i['best'],
                        'text': f"{i['run'] + 1}/{i['n_runs']} 回目（{'・'.join(i['pair'])}）・第 {i['gen']} 世代"})
                return film.search(target, nk, n_env, nk_subst_fn, settings, progress_cb=film_cb, stop_event=stop)
        else:
            def fn(cb, stop):
                return runner.search(fams, target, nk, n_env, nk_subst_fn, settings, progress_cb=cb, stop_event=stop)
        start_job('inv_job_1d', meta, fn)

    render_run('inv_job_1d', 'inv1d', not err and not err2 and settings is not None, start, nk_subst_fn, n_env)

"""2D モードの「🎯 逆設計」のタブ: Mie 共鳴の粒子（誘電体の円柱・球を 1 段、正方・六方格子に並べたもの）を、目標の
色や反射スペクトルから探す。回折が出ない周期に限る（RCWA2D/inverse/families/particles.py）。"""
import numpy as np
import streamlit as st

from RCWA2D.inverse import runner
from RCWA2D.inverse.families.particles import Particles

from . import inverse_app as ia
from .common import build_nk_fn_map

_SHAPE = {'disk': '円柱', 'sphere': '球'}
_LATTICE = {'square': '正方格子', 'hex': '六方格子'}
_SIDE = {'none': '制限しない', 'env': '空気側には出ない（基板側は許す）', 'both': '空気側にも基板側にも出ない'}


def _is_metal(fn):
    """可視域のどこかで Re(n²) < 0（金属）か。"""
    return any((complex(fn(w)) ** 2).real < 0 for w in np.arange(400.0, 781.0, 20.0))


def _settings(nk_namelist, target, n_env, nk_subst_fn):
    mats = st.multiselect('粒子の材料', nk_namelist, key='inv_mie_mats',
                          help='複数選ぶと 1 つずつ全部探す（1 つの構造に使うのは 1 種類）。材料はサイドバーの「材料を追加」で増やせる。'
                               '緑や赤の Mie 共鳴には、シリコンのような高い屈折率が要る。')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        shape = st.radio('粒子の形', list(_SHAPE), format_func=_SHAPE.get, horizontal=True, key='inv_mie_shape',
                         help='目標の色や反射スペクトルに近い Mie 共鳴の粒子の配列（誘電体の円柱・球を基板の上に 1 段）を探す。'
                              '基板と環境の屈折率はサイドバーの設定を使う。計算は S4（φ = 0）。\n\n'
                              '球は薄い円盤に分けて計算するので、円柱より数倍以上時間がかかる。')
        lattice = st.radio('並べ方', list(_LATTICE), format_func=_LATTICE.get, horizontal=True, key='inv_mie_lattice')
        step = ia.step_input()
    with c2:
        p = st.slider('周期 [nm]', 60, 800, step=10, key='inv_mie_p',
                      help='「回折の条件」で制限すると、上限はさらに短くなる。')
        d = st.slider('直径 [nm]', 20, 600, step=10, key='inv_mie_d')
    with c3:
        h = st.slider('高さ [nm]', 10, 800, step=10, key='inv_mie_h', disabled=shape == 'sphere')
        gap = st.number_input('粒子のすき間の最小 [nm]', 1.0, 300.0, step=5.0, key='inv_mie_gap')
    with c4:
        aspect = st.number_input('高さ ÷ 直径 の上限', 0.2, 10.0, step=0.5, key='inv_mie_aspect', disabled=shape == 'sphere')
        side = st.radio('回折の条件', list(_SIDE), format_func=_SIDE.get, key='inv_mie_side',
                        help='制限しない: 周期は上のスライダーの範囲で探す。色は鏡面反射（R0）で評価するので、'
                             '回折が出る周期でも色の計算は正しい。基板側への回折は空気側から見る色に入らない。'
                             '空気側に出る回折光の量は、候補の詳細に出す（見る向きで色が変わる原因になる）。\n\n'
                             'ほかの 2 つは、Mie 共鳴だけで色を出す構造に絞るための条件。'
                             '周期 < 波長 ÷（出る側の屈折率 ＋ sinθ）に限るので、周期の上限がかなり短くなる'
                             '（基板 n=1.5・0/30/60° で約 160 nm）。')
        wl_min = st.number_input('回折を避ける最短波長 [nm]', 380.0, 500.0, step=10.0, key='inv_mie_wl',
                                 disabled=side == 'none',
                                 help='目の感度が低い 380〜430 nm の回折を許すと、周期を少し長くできる。')
    pol, effort, tol = ia.render_common_settings()
    if not mats:
        return mats, None, None, '粒子の材料を 1 つ以上選んでください。'
    nk = build_nk_fn_map(mats)
    metals = [m for m in mats if _is_metal(nk[m])]
    if metals:
        return mats, None, None, f'{"、".join(metals)} は金属なので使えません（Mie 共鳴の逆設計は誘電体だけ）。'
    if target is None:
        return mats, None, None, None
    diff = dict(angles_deg=target.angles, n_env=n_env, nk_sub_fn=nk_subst_fn, side=side, wl_min_nm=float(wl_min))
    try:
        fams = [Particles(m, shape, lattice, p, d, h, gap, aspect, step, diff) for m in mats]
    except ValueError as e:
        return mats, None, None, (f'{e} 屈折率の高い基板では、基板側に回折が出ない周期がとても短くなります。「回折の条件」を'
                                         '「空気側には出ない（基板側は許す）」にするか、角度・基板・直径の下限を見直してください。')
    rng = f'探す周期: {fams[0].variables[0].lo:g}〜{fams[0].variables[0].hi:.0f} nm'
    if side == 'none':
        st.caption(f'回折で周期を制限しない。{rng}')
    else:
        st.caption(f'回折が出ない周期の上限: {fams[0].p_limit:.0f} nm（{"・".join(f"{a:g}°" for a in target.angles)}、'
                   f'{_LATTICE[lattice]}）。{rng}')
    s = runner.Settings(pol=pol, effort=effort, tol=tol)
    ia.time_caption(fams, target, s)
    return mats, fams, s, None


def render_inverse_mie_tab(nk_namelist, n_env, subst_label, nk_subst_fn, blocked=None):
    flash = st.session_state.pop('inv_flash', None)
    if flash:
        st.success(flash)
    ia.collect_job('inv_job_2d', 'inv2d')
    ia.restore_inputs(nk_namelist)
    target, desc, err = ia.render_target()
    st.divider()
    mats, fams, settings, err2 = _settings(nk_namelist, target, n_env, nk_subst_fn)
    ia.remember_inputs()
    if err:
        st.caption(err)
    if err2:
        st.error(err2)

    def start():
        nk = build_nk_fn_map(mats)
        ss = st.session_state
        meta = {**ia.target_meta(target, desc), 'subst': subst_label, 'n_env': float(np.real(n_env)), 'materials': list(mats),
                'settings': vars(settings).copy(), 'step_nm': float(ss['inv_tstep']), 'family': fams[0].name,
                'family_label': f'Mie 共鳴の{_SHAPE[ss["inv_mie_shape"]]}（{_LATTICE[ss["inv_mie_lattice"]]}）'}
        ia.start_job('inv_job_2d', meta, lambda cb, stop: runner.search(fams, target, nk, n_env, nk_subst_fn, settings,
                                                                       progress_cb=cb, stop_event=stop))

    ia.render_run('inv_job_2d', 'inv2d', not err and not err2 and settings is not None, start, nk_subst_fn, n_env, blocked)

"""2Dモードの「収束確認」: 数波長だけ、NumBasis を段階的に増やした計算と、階段近似（球・半球・
横倒しシリンダーの n_slices、レンズ状の凹凸の Δz）を2倍細かくした計算を行い、R0 の変化から
結果が収束しているかを判定する。本計算と同じくバックグラウンドスレッドで解き、停止できる。"""
import threading
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from RCWA2D import s4_parallel

_MAX_NUMBASIS = 801
_POLL_LIMIT = 300   # 0.5秒間隔 × 300 = 約2.5分、これを超えたら手動更新


def default_numbasis_list(n):
    """現在の NumBasis から 1倍・2倍・4倍（上限 801）。"""
    return list(dict.fromkeys(min(int(n) * k, _MAX_NUMBASIS) for k in (1, 2, 4)))


def parse_numbers(text, cast=float):
    """「450, 550, 650」のような入力を数のリストにする。読めないものは無視。"""
    out = []
    for tok in text.replace('、', ',').replace(' ', ',').split(','):
        try:
            out.append(cast(float(tok)))
        except ValueError:
            pass
    return out


def judge(r0_rows):
    """各行（設定ごとの R0、波長方向）について、ひとつ前の行からの最大変化を返す。"""
    return [None] + [float(np.nanmax(np.abs(np.asarray(b) - np.asarray(a)))) for a, b in zip(r0_rows, r0_rows[1:])]


def _worker(runs, wls, theta_rad, args, stop_event, state):
    try:
        for i, (label, kind, layers, nk_fn_map, norder) in enumerate(runs):
            if stop_event.is_set():
                break
            state['current'] = label
            t0 = time.time()
            r = s4_parallel.solve_2d(wls, np.array([theta_rad]), args['phi_rad'], args['pitch_x_um'],
                                     args['pitch_y_um'], norder, layers, nk_fn_map, args['nk_env_fn'],
                                     args['nk_subst_fn'], gamma_deg=args['gamma_deg'], stop_event=stop_event,
                                     pol_deg=args['pol_deg'])
            if not bool(np.all(r['computed'])):
                break
            state['rows'].append(dict(label=label, kind=kind, norder=norder, n_layers=len(layers),
                                      R0=r['R0'][0].tolist(), RT=(r['R_total'][0] + r['T_total'][0]).tolist(),
                                      seconds=time.time() - t0))
    except Exception as e:   # スレッド内の例外は自動伝播しないので明示的に捕まえて表示する
        state['error'] = str(e)


def render_convergence_check(p, calc_args, layers_fn, refined_layers_fn, calc_running):
    """収束確認の UI。refined_layers_fn は階段近似を2倍細かくした (layers, nk_fn_map) を返す関数、
    細かくできる要素がなければ refined_layers_fn=None。"""
    st.markdown("**🧪 収束確認（NumBasis・階段近似の細かさ）**")
    st.caption("数波長だけ、NumBasis を増やした計算と階段近似を2倍細かくした計算を行い、R0 が変わらないかを"
               "確かめる。変化が許容値以下なら、その設定の結果は信頼できる。")
    if 'conv_state' not in st.session_state:
        st.session_state.conv_state = {'thread': None, 'stop_event': None, 'rows': [], 'error': None,
                                       'current': '', 'wls': None, 'tol': 0.005, 'n_runs': 0, 'poll': 0,
                                       't_start': 0.0}
    cs = st.session_state.conv_state
    alive = cs['thread'] is not None and cs['thread'].is_alive()

    mid = 10 * round((p['wl_min_nm'] + p['wl_max_nm']) / 20)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        wl_text = st.text_input("確かめる波長 [nm]（カンマ区切り）",
                                f"{p['wl_min_nm']:.0f}, {mid:.0f}, {p['wl_max_nm']:.0f}", key="conv_wls")
    with c2:
        nb_text = st.text_input("NumBasis の列（カンマ区切り、小さい順）",
                                ", ".join(str(n) for n in default_numbasis_list(p['norder'])), key="conv_nbs",
                                help=f"既定はサイドバーの NumBasis の1・2・4倍（上限 {_MAX_NUMBASIS}）。"
                                     "NumBasis 401 以上は1点に数分かかることがある。")
    with c3:
        tol = st.number_input("許容する R0 の変化", 0.0001, 0.1, 0.005, 0.001, format="%.4f", key="conv_tol")
    refine = refined_layers_fn is not None
    do_slices = st.checkbox("階段近似を2倍細かくした計算も行う（サイドバーの NumBasis で）", value=refine,
                            disabled=not refine, key="conv_slices",
                            help="球・半球・横倒しシリンダーの n_slices を2倍、レンズ状の凹凸の Δz を半分にする。"
                                 + ("" if refine else " いまの構造には階段近似の要素がない。"))
    theta_deg = float(p['theta_deg'])
    st.caption(f"入射角 θ = {theta_deg:.0f}°、φ = {p['phi_deg']:.0f}°、偏光角 ψ = {p['pol_deg']:.0f}°"
               "（サイドバーの設定。角度掃引では最初の角度）。")

    b1, b2 = st.columns(2)
    with b1:
        start = st.button("🧪 収束を確認", disabled=alive or calc_running, use_container_width=True,
                          help="本計算の実行中は使えない。" if calc_running else None)
    with b2:
        stop = st.button("⏹ 収束確認を停止", disabled=not alive, use_container_width=True, key="conv_stop")

    if start:
        wls = np.array(sorted(w for w in parse_numbers(wl_text) if 200.0 <= w <= 2000.0))
        nbs = sorted(dict.fromkeys(n for n in parse_numbers(nb_text, int) if 9 <= n <= _MAX_NUMBASIS))
        if len(wls) == 0 or len(nbs) == 0:
            st.error("波長と NumBasis を1つ以上入力してください（NumBasis は 9〜801）。")
            return
        layers, nk_fn_map = layers_fn()
        norder = int(p['norder'])
        if do_slices and refine:   # 比べる相手（同じ NumBasis・元の細かさ）も必ず解く
            nbs = sorted(set(nbs) | {norder})
        runs = [(f"NumBasis {n}", 'nb', layers, nk_fn_map, n) for n in nbs]
        if do_slices and refine:
            r_layers, r_map = refined_layers_fn()
            runs.append((f"NumBasis {norder}・階段2倍細かく", 'slice', r_layers, r_map, norder))
        stop_event = threading.Event()
        cs.update(stop_event=stop_event, rows=[], error=None, current='', wls=wls, tol=float(tol),
                  n_runs=len(runs), poll=0, t_start=time.time(), norder=int(p['norder']))
        th = threading.Thread(target=_worker, daemon=True,
                              args=(runs, wls, np.radians(theta_deg), calc_args, stop_event, cs))
        cs['thread'] = th
        th.start()
        st.rerun()
    if stop and cs['stop_event'] is not None:
        cs['stop_event'].set()
        st.warning("停止をリクエストしました。実行中の設定が終わり次第止まります。")

    if cs['error']:
        st.error(f"収束確認中にエラーが発生しました: {cs['error']}")
    _render_rows(cs)

    if alive:
        done = len(cs['rows'])
        st.progress(done / cs['n_runs'] if cs['n_runs'] else 0.0,
                    text=f"{done}/{cs['n_runs']} 設定  計算中: {cs['current']}  経過 {time.time() - cs['t_start']:.0f}秒")
        if cs['poll'] < _POLL_LIMIT:
            cs['poll'] += 1
            time.sleep(0.5)
            st.rerun()
        elif st.button("🔄 収束確認の状態を更新"):
            cs['poll'] = 0
            st.rerun()


def _render_rows(cs):
    rows = cs['rows']
    if not rows:
        return
    wls, tol = cs['wls'], cs['tol']
    nb_rows = [r for r in rows if r['kind'] == 'nb']
    sl_rows = [r for r in rows if r['kind'] == 'slice']

    table = []
    nb_changes = judge([r['R0'] for r in nb_rows])
    for r, ch in zip(nb_rows, nb_changes):
        table.append(_table_row(r, wls, ch))
    sl_change = None
    if sl_rows:
        base = next((r for r in nb_rows if r['norder'] == sl_rows[0]['norder']), None)
        if base is not None:
            sl_change = float(np.nanmax(np.abs(np.asarray(sl_rows[0]['R0']) - np.asarray(base['R0']))))
        table.append(_table_row(sl_rows[0], wls, sl_change))
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("max|ΔR0|: NumBasis の行はひとつ上の行との差、「階段2倍細かく」の行は同じ NumBasis の元の細かさとの差"
               "（確かめた波長のうち最大）。")

    # 判定
    if len(nb_rows) >= 2:
        last = nb_changes[-1]
        a, b = nb_rows[-2]['norder'], nb_rows[-1]['norder']
        if last <= tol:
            st.success(f"✅ NumBasis {a} → {b} で R0 の変化は最大 {last:.4f}（許容 {tol:.4f} 以下）。"
                       f"NumBasis {a} 以上なら収束しています。")
        else:
            st.warning(f"⚠ NumBasis {a} → {b} で R0 が最大 {last:.4f} 変わります（許容 {tol:.4f} 超）。"
                       f"まだ収束していません。NumBasis をさらに増やして確かめてください。")
    if sl_change is not None:
        n = sl_rows[0]['norder']
        if sl_change <= tol:
            st.success(f"✅ 階段近似を2倍細かくしても R0 の変化は最大 {sl_change:.4f}（NumBasis {n}）。"
                       "いまの n_slices・Δz で十分です。")
        else:
            st.warning(f"⚠ 階段近似を2倍細かくすると R0 が最大 {sl_change:.4f} 変わります（NumBasis {n}）。"
                       "n_slices を増やす（Δz を小さくする）必要があります。")
    rt_bad = [r['label'] for r in rows if np.nanmax(np.abs(np.asarray(r['RT']) - 1.0)) > 0.01]
    if rt_bad:
        st.info("R+T が 1 から 1% 以上ずれている設定があります（吸収のある材質なら正常。吸収がないなら"
                f"次数不足のしるし）: {', '.join(rt_bad)}")

    if len(nb_rows) >= 2:
        fig = go.Figure()
        x = [r['norder'] for r in nb_rows]
        for j, wl in enumerate(wls):
            fig.add_trace(go.Scatter(x=x, y=[r['R0'][j] for r in nb_rows], mode='lines+markers', name=f"{wl:.0f} nm"))
        fig.update_layout(height=320, margin=dict(l=50, r=10, t=40, b=40), title="R0 と NumBasis",
                          xaxis_title="NumBasis", yaxis_title="R0", plot_bgcolor='white')
        fig.update_xaxes(gridcolor="#d5dbe1"); fig.update_yaxes(gridcolor="#d5dbe1")
        st.plotly_chart(fig, use_container_width=True)


def _table_row(r, wls, change):
    row = {'設定': r['label'], 'S4の層数': r['n_layers']}
    for wl, v in zip(wls, r['R0']):
        row[f'R0 {wl:.0f}nm'] = round(float(v), 4)
    row['max|ΔR0|'] = None if change is None else round(change, 4)   # 空欄（文字を混ぜると表の型が崩れる）
    row['R+T（範囲）'] = f"{min(r['RT']):.4f}〜{max(r['RT']):.4f}"
    row['時間 [秒]'] = round(r['seconds'], 1)
    return row

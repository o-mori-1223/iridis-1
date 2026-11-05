"""S4計算（1D列構造・2D周期アレイ）を、常駐ワーカープロセスで並列に解く。

RCWAの各(波長, 入射角)点は互いに独立なので、点をチャンク（同じ入射角・連続した
波長の塊）に分けて ProcessPoolExecutor に投げる。ワーカーはチャンク内で S4 の
sim を1つ作り回して解く（rcwa_s4_backend / rcwa_s4_array2d_backend の既存関数を
そのまま呼ぶ——計算ロジックは無変更）。

実測（WSL, 8論理コア, 2026-10-26）: OpenBLASを1スレッドに固定した上で、直列比
約2.2〜2.4倍（1D: 6列DBR nOrder=43、2D: 円盤 NumBasis=101、各64点）。
OpenBLASのスレッド並列は行列が小さいため逆効果で、1スレッド固定だけでも
1.6〜1.7倍速くなる（s4lab.py の読み込み時に設定）。

- ワーカーは spawn で起動し、プロセスは使い回す（初回だけ起動・import の数秒がかかる）。
  ワーカー数は環境変数 IRIDIS_WORKERS（既定 os.cpu_count()）。1 なら並列化せず
  呼び出し元プロセスで直列に解く。
- 材質の n,k 関数（interp1d やラムダ）はプロセス間で渡せないので、呼び出し側で
  全波長点の値を前もって標本化し、ワーカー側で「波長→値」の表引き関数に戻す。
- Streamlit には依存しない（UI側は progress_cb / stop_event で進捗表示・停止する）。
"""
import atexit
import math
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import numpy as np

_executor = None
_executor_lock = threading.Lock()


def n_workers():
    try:
        return max(1, int(os.environ.get('IRIDIS_WORKERS', '')))
    except ValueError:
        return max(1, os.cpu_count() or 1)


def _get_executor():
    """常駐プールを返す。ワーカーが異常終了してプールが壊れていたら作り直す。
    注意: spawn のワーカーは親の __main__ を __mp_main__ として読み込み直すので、
    呼び出し元のスクリプトは `if __name__ == '__main__':` ガードを持つこと（iridis.py 参照）。"""
    global _executor
    with _executor_lock:
        if _executor is not None and getattr(_executor, '_broken', False):
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None
        if _executor is None:
            _executor = ProcessPoolExecutor(n_workers(), mp_context=mp.get_context('spawn'))
            atexit.register(_executor.shutdown, wait=False, cancel_futures=True)
        return _executor


def _warm_task():
    """両backendを import して S4本体(libs4lab.so)の読み込みまで済ませる。"""
    from RCWA2D import rcwa_s4_array2d_backend, rcwa_s4_backend
    return rcwa_s4_backend.HAVE_S4 and rcwa_s4_array2d_backend.HAVE_S4


def warm_up():
    """ワーカーを先に起動しておく（待たずに返る）。ワーカーの起動・import には数秒かかる
    ので、アプリ起動時に呼んでおくと最初の計算でその待ち時間が出ない。"""
    if n_workers() > 1:
        ex = _get_executor()
        for _ in range(n_workers()):
            ex.submit(_warm_task)


# ---------------------------------------------------------------- n,k の受け渡し
def _sample(fn, wls):
    return np.array([complex(fn(float(w))) for w in wls], dtype=complex)


def _lookup_fn(wls, values):
    table = {float(w): complex(v) for w, v in zip(wls, values)}
    return lambda wl: table[float(wl)]


def _sample_map(nk_fn_map, wls):
    return {name: _sample(fn, wls) for name, fn in nk_fn_map.items()}


def _lookup_map(wls, samples):
    return {name: _lookup_fn(wls, v) for name, v in samples.items()}


# ---------------------------------------------------------------- ワーカー本体
def _solve_1d_chunk(args):
    from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
    wls, angle_rad, pitch_um, norder, layers, nk_samples, n_env, subst_samples = args
    return calc_rcwa1d_s4(np.asarray(wls), angle_rad, pitch_um, norder, layers,
                          _lookup_map(wls, nk_samples), n_env, _lookup_fn(wls, subst_samples))


def _solve_2d_chunk(args):
    from RCWA2D.rcwa_s4_array2d_backend import calc_array_rcwa_s4
    (wls, theta_rad, phi_rad, pitch_x_um, pitch_y_um, norder, layers, nk_samples,
     env_samples, subst_samples, gamma_deg, pol_deg) = args
    r = calc_array_rcwa_s4(np.asarray(wls), theta_rad, phi_rad, pitch_x_um, pitch_y_um, norder,
                           layers, _lookup_map(wls, nk_samples), _lookup_fn(wls, env_samples),
                           _lookup_fn(wls, subst_samples), gamma_deg=gamma_deg, pol_deg=pol_deg)
    return r['R0'], r['R_total'], r['T0'], r['T_total'], r['order_basis']


# ---------------------------------------------------------------- 実行・チャンク分割
def _chunks(n_angles, n_wl, workers):
    """(角度index, 波長slice) のリスト。ワーカー数の約4倍のタスクになるよう波長を分割する
    （負荷分散のため）。同じチャンク内では S4 の sim を使い回せるので細かくしすぎない。"""
    per_angle = max(1, math.ceil(4 * workers / max(1, n_angles)))
    size = max(1, math.ceil(n_wl / per_angle))
    return [(ia, slice(s, min(s + size, n_wl)))
            for ia in range(n_angles) for s in range(0, n_wl, size)]


def _mem_available_bytes():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def estimate_task_bytes(n_layers, norder):
    """S4 の1タスク（1つの sim）が使うメモリの概算。各層の固有モード（φ, kp, ε^-1 など）と
    一括ソルバーの作業領域が、層ごとに (2·NumBasis)^2 の複素行列 約9枚分。
    実測: NumBasis=201・約60層を8並列で解くと WSL（7.8GB）が落ちた（2026-10-26）。"""
    n2 = 2 * max(1, int(norder))
    return int(max(1, n_layers) * 9 * n2 * n2 * 16)


def max_concurrency(n_layers, norder):
    """空きメモリの7割に収まる同時実行数（1〜ワーカー数）。"""
    avail = _mem_available_bytes()
    if avail is None:
        return n_workers()
    return max(1, min(n_workers(), int(0.7 * avail // estimate_task_bytes(n_layers, norder))))


def _run(fn, tasks, progress_cb=None, stop_event=None, weights=None, max_inflight=None):
    """tasks を並列に解き、結果のリストを返す（停止・中断で解かなかったタスクは None）。
    progress_cb(done, total) は1タスク完了ごとに呼ぶ（weights があればその合計で数える）。
    progress_cb が例外を投げた場合（Streamlit の再実行による中断など）も、未着手の
    タスクは取り消してから例外をそのまま伝える。同時に投げるタスクは max_inflight 個まで
    （メモリ不足で WSL ごと落ちるのを防ぐ）。"""
    weights = weights or [1] * len(tasks)
    total, done = sum(weights), 0
    results = [None] * len(tasks)
    if n_workers() == 1 or max_inflight == 1:
        for i, t in enumerate(tasks):
            if stop_event is not None and stop_event.is_set():
                break
            results[i] = fn(t)
            done += weights[i]
            if progress_cb is not None:
                progress_cb(done, total)
        return results

    ex = _get_executor()
    limit = max_inflight or len(tasks)
    queue = list(range(len(tasks)))
    futures, pending = {}, set()

    def submit_more():
        while queue and len(pending) < limit:
            i = queue.pop(0)
            f = ex.submit(fn, tasks[i])
            futures[f] = i
            pending.add(f)

    try:
        submit_more()
        while pending:
            finished, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for f in finished:
                i = futures[f]
                results[i] = f.result()
                done += weights[i]
            if finished and progress_cb is not None:
                progress_cb(done, total)
            if stop_event is not None and stop_event.is_set():
                break
            submit_more()
    finally:
        for f in pending:
            f.cancel()
    return results


def solve_1d(wl_nm_ar, angles_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn,
             progress_cb=None):
    """1D列構造を複数の入射角で解く。戻り値は角度ごとの (irp, itp, irs, its) のリスト
    （各 [nwl, norder]、rcwa_s4_backend.calc_rcwa1d_s4 と同じ形・同じ次数の並び）。"""
    wls = np.asarray(wl_nm_ar, dtype=float)
    nk_samples, subst_samples = _sample_map(nk_fn_map, wls), _sample(nk_subst_fn, wls)
    plan = _chunks(len(angles_rad), len(wls), n_workers())
    tasks = [(wls[sl], float(angles_rad[ia]), pitch_um, norder, layers,
              {k: v[sl] for k, v in nk_samples.items()}, n_env, subst_samples[sl])
             for ia, sl in plan]
    parts = _run(_solve_1d_chunk, tasks, progress_cb, weights=[sl.stop - sl.start for _, sl in plan],
                 max_inflight=max_concurrency(len(layers) + 2, norder))

    out = [tuple(np.empty((len(wls), norder)) for _ in range(4)) for _ in angles_rad]
    for (ia, sl), part in zip(plan, parts):
        for dst, src in zip(out[ia], part):
            dst[sl] = src
    return out


def solve_2d(wl_nm_ar, theta_rad_ar, phi_rad, pitch_x_um, pitch_y_um, norder, layers,
             nk_fn_map, nk_env_fn, nk_subst_fn, gamma_deg=90.0, stop_event=None, progress_cb=None,
             pol_deg=0.0):
    """2D周期アレイを複数の入射角で解く。戻り値は
    rcwa_s4_array2d_backend.calc_array_rcwa_s4_angle_sweep と同じ形の dict
    （'R0','R_total','T0','T_total' が [n_theta, nwl]、未計算点は NaN、
    'order_basis'、'n_computed'、計算済みの点を示す bool 配列 'computed'）。
    stop_event がセットされたら未着手の点を取り消して返す。
    pol_deg: 偏光角 ψ（0 = s 偏光、90 = p 偏光、その間は直線偏光。calc_array_rcwa_s4 参照）。"""
    wls = np.asarray(wl_nm_ar, dtype=float)
    nk_samples = _sample_map(nk_fn_map, wls)
    env_samples, subst_samples = _sample(nk_env_fn, wls), _sample(nk_subst_fn, wls)
    plan = _chunks(len(theta_rad_ar), len(wls), n_workers())
    tasks = [(wls[sl], float(theta_rad_ar[ia]), phi_rad, pitch_x_um, pitch_y_um, norder, layers,
              {k: v[sl] for k, v in nk_samples.items()}, env_samples[sl], subst_samples[sl], gamma_deg,
              float(pol_deg))
             for ia, sl in plan]
    parts = _run(_solve_2d_chunk, tasks, progress_cb, stop_event,
                 weights=[sl.stop - sl.start for _, sl in plan],
                 max_inflight=max_concurrency(len(layers) + 2, norder))

    shape = (len(theta_rad_ar), len(wls))
    out = {k: np.full(shape, np.nan) for k in ('R0', 'R_total', 'T0', 'T_total')}
    computed = np.zeros(shape, dtype=bool)
    basis = None
    for (ia, sl), part in zip(plan, parts):
        if part is None:
            continue
        for k, v in zip(('R0', 'R_total', 'T0', 'T_total'), part[:4]):
            out[k][ia, sl] = v
        basis = part[4]
        computed[ia, sl] = True
    out.update(order_basis=basis, n_computed=int(computed.sum()), computed=computed)
    return out

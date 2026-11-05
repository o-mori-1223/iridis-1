"""1Dモードの計算: S4（RCWA2D.rcwa_s4_backend）を入射角ごとにキャッシュしつつ、
未計算の角度だけをまとめて RCWA2D.s4_parallel でプロセス並列に解く。

キャッシュは（構造・ピッチ・nOrder・環境屈折率・波長グリッド・全材質のn,k標本・入射角）
をキーにした、プロセス内で共有の LRU。単一角度で計算した角度は、その後の角度掃引や
色スウォッチ(0/30/60°)でも再利用される。
"""
import threading
import time
from collections import OrderedDict

import numpy as np

from RCWA2D import s4_parallel

_CACHE = OrderedDict()
_CACHE_MAX = 256           # 1エントリ = 4 x [nwl, norder] の float 配列
_CACHE_LOCK = threading.Lock()


def layers_cache_key(layers):
    """layers（list of {'thickness', 'segments'}）→ ハッシュ可能なタプル。"""
    return tuple((L['thickness'], tuple((m, round(w, 12)) for m, w in L['segments']))
                 for L in layers)


def _nk_key(nk_fn_map, nk_subst_fn, wls):
    def samp(fn):
        return tuple((round(c.real, 12), round(c.imag, 12)) for c in (complex(fn(float(w))) for w in wls))
    return (tuple((name, samp(nk_fn_map[name])) for name in sorted(nk_fn_map)), samp(nk_subst_fn))


def calc_1d(wl_nm_ar, angles_deg, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn,
            progress_cb=None):
    """入射角（度）のリストそれぞれについて (irp, itp, irs, its)（各 [nwl, norder]）を返す。
    戻り値は {angle_deg: (irp, itp, irs, its)}。progress_cb(done, total) は未計算分の
    並列計算の進捗（波長点数単位）。"""
    wls = np.asarray(wl_nm_ar, dtype=float)
    base = (layers_cache_key(layers), float(pitch_um), int(norder), float(n_env),
            tuple(wls.tolist()), _nk_key(nk_fn_map, nk_subst_fn, wls))
    angles = [float(a) for a in dict.fromkeys(float(a) for a in angles_deg)]

    out, missing = {}, []
    with _CACHE_LOCK:
        for a in angles:
            key = base + (a,)
            if key in _CACHE:
                _CACHE.move_to_end(key)
                out[a] = _CACHE[key]
            else:
                missing.append(a)

    if missing:
        t0 = time.time()
        results = s4_parallel.solve_1d(wls, np.radians(missing), pitch_um, norder, layers,
                                       nk_fn_map, n_env, nk_subst_fn, progress_cb=progress_cb)
        print(f'S4 1D: {len(missing)} angle(s) x {len(wls)} wl in {time.time() - t0:.2f} s '
              f'({s4_parallel.n_workers()} workers)')
        with _CACHE_LOCK:
            for a, r in zip(missing, results):
                out[a] = r
                _CACHE[base + (a,)] = r
                while len(_CACHE) > _CACHE_MAX:
                    _CACHE.popitem(last=False)
    return out

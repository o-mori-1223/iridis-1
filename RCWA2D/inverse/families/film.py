"""平らな多層膜の逆設計: 目標の色（sRGB、入射角ごと）または反射スペクトルに近い膜を探す。

構造: 環境 / 材料 A・B の交互の膜（最大 n_max_layers 層、上から A, B, A, …）/ 基板。
使う材料は 2 種類まで。候補の材料が 3 つ以上なら、2 つずつの組み合わせを順に全部探す。
各層の厚さは 0〜t_max_nm で別々に探し、t_min_nm より薄い層は「無い」とみなす（0 nm にする）。
これで層の数（と、どちらの材料が一番上・一番下に来るか）も探索の中で決まる。
見つける膜厚は t_step_nm（既定 1 nm）の整数倍にする。表示する色・ずれは、刻みに乗せた膜厚そのもので計算した値。

順計算は伝達行列法（RCWA2D/tmm.py、S4 と一致することを tests/inverse_check.py で確認）。
最適化は scipy の差分進化（differential_evolution、1 世代の候補をまとめて計算する vectorized）のあと、
刻みの上での近傍探索（_refine）で仕上げる。刻みが _CONTINUOUS_STEP_NM 以下なら、差分進化は刻みなしの膜厚で行う
（1〜10 nm ではこの方が良い膜が見つかりやすい）。それより粗い刻みでは、はじめから刻みの上で行う。
Streamlit には依存しない（画面は progress_cb と stop_event で進捗表示・停止する）。
"""
import itertools
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import differential_evolution

from ... import color_metrics as cm
from ... import tmm
from .. import targets


@dataclass
class SearchSettings:
    n_max_layers: int = 8
    t_min_nm: float = 20.0       # これより薄い層は作らない（無い層になる）
    t_max_nm: float = 300.0      # 1 層の厚さの上限
    t_step_nm: float = 1.0       # 膜厚の刻み（膜厚はこの整数倍にする。0 なら刻みなし）
    total_max_nm: float = 0.0    # 総膜厚の上限（0 なら制限なし）
    pol: str = 'avg'             # 'avg'（偏光していない光）/ 's' / 'p'
    effort: str = 'standard'
    tol: float = 0.0             # 損失がこれ以下になったらその組み合わせの探索を打ち切る（0 なら打ち切らない）
    seed: int = 0


# 探索の細かさ: 世代数・1 世代の候補数（変数 1 つあたり）・初期値を変えて走らせる回数
EFFORTS = {
    'quick': dict(maxiter=150, popsize=10, restarts=1),
    'standard': dict(maxiter=400, popsize=15, restarts=2),
    'careful': dict(maxiter=1000, popsize=20, restarts=3),
}
_MAX_SPECTRUM_POINTS = 121
_CONTINUOUS_STEP_NM = 10.0
_TOTAL_PENALTY_PER_NM = 0.05   # 総膜厚の上限を超えた 1 nm あたりの罰則（ΔE か RMSE[%] の単位）


def material_pairs(names):
    """候補の材料 → 探す組み合わせ（2 つずつ。1 つだけなら単一の材料）。"""
    names = list(dict.fromkeys(names))
    if len(names) <= 1:
        return [tuple(names)]
    return list(itertools.combinations(names, 2))


def stack_materials(pair, n_layers):
    """上から A, B, A, … の材料の並び。"""
    return [pair[i % len(pair)] for i in range(n_layers)]


def max_on_grid(t_max_nm, t_step_nm):
    """t_max 以下で一番厚い、刻みの整数倍の膜厚（刻みなしなら t_max）。"""
    if t_step_nm <= 0:
        return t_max_nm
    return round(np.floor(t_max_nm / t_step_nm + 1e-9) * t_step_nm, 6)


def snap(d, s, n_mats=2):
    """膜厚 [候補, 層]（または [層]）を刻みに乗せ、最小膜厚より薄い層は 0 nm にする。
    間の層が 0 nm になって同じ材料の層がつながるときは、つながった厚さが 1 層の最大膜厚を超えないように
    後の（下の）層を薄くする。n_mats: 材料の数（層の材料は上から n_mats 種類の繰り返し）。"""
    d = np.asarray(d, float)
    if s.t_step_nm > 0:
        d = np.round(np.round(d / s.t_step_nm) * s.t_step_nm, 6)
    t_max = max_on_grid(s.t_max_nm, s.t_step_nm)
    d = np.where(d < s.t_min_nm - 1e-9, 0.0, np.minimum(d, t_max))
    D = np.atleast_2d(d).copy()
    last = np.full(D.shape[0], -1)          # 直前の（0 nm でない）層の材料
    run = np.zeros(D.shape[0])              # その材料がつながっている厚さ
    for j in range(D.shape[1]):
        m = j % n_mats
        here = D[:, j] > 0
        joined = np.where(last == m, run, 0.0)
        D[:, j] = np.where(here, np.minimum(D[:, j], np.round(t_max - joined, 6)), 0.0)
        run = np.where(here, joined + D[:, j], run)
        last = np.where(here, m, last)
    return D.reshape(d.shape)


def compact_layers(mats, d_nm):
    """0 nm の層を除き、隣り合う同じ材料の層をつなげる → [(材料, 厚さ nm), …]（上から）。"""
    out = []
    for m, t in zip(mats, d_nm):
        if t <= 0:
            continue
        if out and out[-1][0] == m:
            out[-1] = (m, round(out[-1][1] + float(t), 6))
        else:
            out.append((m, float(t)))
    return out


def _sample(fn, wl):
    return np.array([complex(fn(w)) for w in wl])


class _Problem:
    """1 つの材料の組み合わせについて、膜厚の組 [候補, 層] → 損失 [候補] を計算する。"""

    def __init__(self, target, pair, materials, n_env, nk_sub_fn, settings):
        self.target, self.s = target, settings
        self.mats = stack_materials(pair, settings.n_max_layers)
        self.n_env = float(np.real(n_env))
        self.n_mats = len(pair)
        self.angles = target.angles
        self.wl = targets.wavelengths(target, 1, _MAX_SPECTRUM_POINTS)
        self.n_stack = np.array([_sample(materials[m], self.wl) for m in self.mats])
        self.n_sub = _sample(nk_sub_fn, self.wl)

    def spectra(self, D):
        """[候補, 層] → 角度ごとの反射率のリスト（各 [候補, 波長]）。"""
        return [tmm.reflectance_unpolarized(self.n_stack, D, self.wl, th, self.n_env, self.n_sub, self.s.pol)
                for th in self.angles]

    def loss_parts(self, D):
        """[候補, 層]（t_min 処理済み）→（損失 [候補], 角度ごとの値 [候補, 角度]）。"""
        main, per = targets.losses(self.target, self.wl, self.spectra(D))
        if self.s.total_max_nm > 0:
            main = main + _TOTAL_PENALTY_PER_NM * np.maximum(D.sum(axis=1) - self.s.total_max_nm, 0.0)
        return main, per

    def __call__(self, x):
        """differential_evolution の目的関数（vectorized: x は [層, 候補]、1 次元なら 1 候補）。"""
        x = np.asarray(x, float)
        single = x.ndim == 1
        D = snap(x[None, :] if single else x.T, self.s, self.n_mats)
        loss = self.loss_parts(D)[0]
        return float(loss[0]) if single else loss


def _describe(problem, pair, d):
    """膜厚の組 1 つ → 候補の情報（画面と記録に出すもの）。"""
    d = snap(d, problem.s, problem.n_mats)
    loss, per = problem.loss_parts(d[None, :])
    Rs = problem.spectra(d[None, :])
    layers = compact_layers(problem.mats, d)
    cand = {
        'family': 'film', 'pair': list(pair), 'layers': layers, 'loss': float(loss[0]),
        'total_nm': round(float(sum(t for _, t in layers)), 6), 'n_layers': len(layers),
        'angles_deg': list(problem.angles), 'per_angle': [float(v) for v in per[0]],
        'hexes': targets.hexes_of(problem.wl, [R[0] for R in Rs]),
        'wl_nm': problem.wl.tolist(), 'R': [R[0].tolist() for R in Rs],
    }
    return cand


def _distinct(cands, tol_nm=3.0):
    """同じ材料・同じ層の並びで、厚さがどれも tol_nm 以内の候補は、良い方だけ残す。"""
    out = []
    for c in sorted(cands, key=lambda c: c['loss']):
        dup = any(o['pair'] == c['pair'] and [m for m, _ in o['layers']] == [m for m, _ in c['layers']]
                  and all(abs(a - b) <= tol_nm for (_, a), (_, b) in zip(o['layers'], c['layers']))
                  for o in out)
        if not dup:
            out.append(c)
    return out


def _refine(problem, d, max_iter=200):
    """刻みに乗った膜厚の組 d から、損失が小さくなる近くの組へ移ることを、良くならなくなるまで繰り返す。
    近くの組: 1 層を ±1〜10 刻み動かす・1 層を無くす・2 層を ±1 刻みずつ同時に動かす。"""
    s = problem.s
    step = s.t_step_nm
    d = snap(d, s, problem.n_mats)
    best = float(problem.loss_parts(d[None, :])[0][0])
    n = d.size
    unit = np.eye(n) * step
    moves = [k * unit for k in (1, -1, 2, -2, 3, -3, 5, -5, 10, -10)]
    iu, ju = np.triu_indices(n, 1)
    moves += [a * unit[iu] + b * unit[ju] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    moves = np.concatenate(moves)
    for _ in range(max_iter):
        trial = np.concatenate([d + moves, np.where(np.eye(n, dtype=bool), 0.0, d)])
        trial = snap(np.clip(trial, 0.0, None), s, problem.n_mats)
        loss = problem.loss_parts(trial)[0]
        k = int(np.argmin(loss))
        if not loss[k] < best - 1e-9:
            break
        d, best = trial[k], float(loss[k])
    return d


def _make_callback(run_i, n_runs, pair, maxiter, tol, progress_cb, stop_event):
    """世代ごとの callback。scipy は引数の名前が intermediate_result のときだけ OptimizeResult を渡すので、
    引数はその 1 つだけにする。True を返すと打ち切る。"""
    state = {'gen': 0}

    def callback(intermediate_result):
        state['gen'] += 1
        best = float(intermediate_result.fun)
        if progress_cb is not None:
            progress_cb({'run': run_i, 'n_runs': n_runs, 'pair': pair, 'gen': state['gen'],
                         'maxiter': maxiter, 'best': best})
        return bool((stop_event is not None and stop_event.is_set()) or (tol > 0 and best <= tol))

    return callback


def search(target, materials, n_env, nk_sub_fn, settings=None, progress_cb=None, stop_event=None):
    """目標に近い多層膜を探す。

    materials: {材料名: 波長[nm] → 複素屈折率 の関数}（候補の材料。2 つずつの組み合わせを全部探す）。
    progress_cb(info): 世代ごとに呼ぶ。info = {'run', 'n_runs', 'pair', 'gen', 'maxiter', 'best'}。
    stop_event: threading.Event。立つと探索を打ち切り、そこまでの結果を返す。
    戻り値: 候補のリスト（損失の小さい順。重なりは除く）。
    """
    s = settings or SearchSettings()
    eff = EFFORTS[s.effort]
    pairs = material_pairs(list(materials))
    runs = [(p, r) for p in pairs for r in range(eff['restarts'])]
    cands = []
    for run_i, (pair, restart) in enumerate(runs):
        if stop_event is not None and stop_event.is_set():
            break
        problem = _Problem(target, pair, materials, n_env, nk_sub_fn, s)
        continuous = s.t_step_nm <= _CONTINUOUS_STEP_NM
        de_problem = (_Problem(target, pair, materials, n_env, nk_sub_fn, replace(s, t_step_nm=0.0))
                      if continuous and s.t_step_nm > 0 else problem)
        res = differential_evolution(
            de_problem, [(0.0, s.t_max_nm)] * s.n_max_layers, maxiter=eff['maxiter'], popsize=eff['popsize'],
            tol=1e-4, mutation=(0.5, 1.0), recombination=0.7, init='sobol',
            rng=np.random.default_rng(s.seed + 1000 * run_i), vectorized=True, updating='deferred',
            polish=continuous, callback=_make_callback(run_i, len(runs), pair, eff['maxiter'], s.tol, progress_cb, stop_event))
        stopped = stop_event is not None and stop_event.is_set()
        on_grid = (lambda d: d) if s.t_step_nm <= 0 or stopped else (lambda d: _refine(problem, d))
        cands.append(_describe(problem, pair, on_grid(res.x)))
        # 最終世代の中から、別の形の良い候補も拾う（同じ色になる別の膜を見せるため）
        pop, energies = getattr(res, 'population', None), getattr(res, 'population_energies', None)
        if pop is not None and energies is not None:   # population は膜厚 [nm] そのもの
            for k in np.argsort(energies)[:5]:
                cands.append(_describe(problem, pair, on_grid(pop[k])))
    return _distinct(cands)


def structure_spectra(layers, materials, n_env, nk_sub_fn, wl_nm, angles_deg, pol='avg'):
    """[(材料, 厚さ), …]（上から）の膜の反射率 → 角度ごとの [波長] のリスト（詳細の図用）。"""
    wl = np.asarray(wl_nm, float)
    n_stack = np.array([_sample(materials[m], wl) for m, _ in layers]).reshape(len(layers), wl.size)
    d = np.array([[t for _, t in layers]])
    n_sub = _sample(nk_sub_fn, wl)
    return [tmm.reflectance_unpolarized(n_stack, d, wl, th, float(np.real(n_env)), n_sub, pol)[0]
            for th in angles_deg]

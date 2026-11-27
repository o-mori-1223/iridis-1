"""S4 を使う構造（1D の格子・列構造、Mie 共鳴の粒子の配列）の逆設計の探索。Streamlit 非依存。

S4 は次数（NumBasis）を増やすほど正確だが急に重くなる。そこで 3 段階で探す。
  1. 広く探す   粗い次数で、一様にばらまいた候補 → 良い候補のまわりをくり返し探す
  2. 寸法を直す 良かった候補を中くらいの次数で計算し直し、そのまわりを探して、傾きを使って仕上げる
                （粗い次数の最適値は正しくないため）
  3. 確かめる   上位の候補を細かい次数・細かい波長で計算し直す。2 段目との差を補正して寸法を直し、もう一度確かめる。
                結果に出すのは、この細かい計算の値
候補（構造）1 つを 1 つのタスクとして、s4_parallel の常駐ワーカーで並列に解く。候補がワーカーより少ないときは
候補を角度・波長ごとに分けて解く。同時に解く数は、空きメモリに収まる数まで（s4_parallel.max_concurrency）。
寸法は刻みの整数倍だけを候補にする。

構造の種類（family）は次を持つオブジェクト（families/*.py）:
  kind                '1d' または '2d'（使う S4 の計算）
  variables           [Var, …]（探す寸法）
  levels              [Level, …] × 3（段階ごとの次数など）
  materials           使う材料の名前
  project(X)          [候補, 変数] を刻みに乗せ、作れる範囲へ直す
  geometry(x, level)  寸法 → {'pitch_nm', 'gamma_deg', 'layers'}
  describe(x)         寸法 → {'summary', 'dims', 'apply'}（画面の表示と、編集画面へ渡す構造）
  post(ctx, x, level, solve)  （任意）確かめた候補への追加の情報
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .. import s4_parallel
from . import targets


@dataclass(frozen=True)
class Var:
    key: str
    label: str
    lo: float
    hi: float
    step: float = 1.0
    unit: str = 'nm'


@dataclass(frozen=True)
class Level:
    norder: int
    n_slices: int = 0        # 球の分割数（使う構造だけ）
    cie_stride: int = 4      # 色のときの波長の刻み = 5·cie_stride nm
    n_wl_max: int = 21       # スペクトルのときの波長の点数の上限


# 候補数。n0: ばらまく数、r1: (回数, 1 回の候補数)、top2: 2 段目で計算し直す数、r2: (回数, 候補数)、
# polish: 仕上げ（_polish）の回数、verify: 確かめる数、
# correct: 細かい計算との差を補正して直す回数
EFFORTS = {
    'quick': dict(n0=64, r1=(3, 24), top2=6, r2=(1, 12), polish=3, verify=2, correct=1),
    'standard': dict(n0=192, r1=(4, 32), top2=12, r2=(2, 20), polish=4, verify=3, correct=2),
    'careful': dict(n0=384, r1=(6, 48), top2=16, r2=(4, 24), polish=7, verify=5, correct=3),
}


@dataclass
class Settings:
    pol: str = 'avg'         # 'avg'（偏光していない光）/ 's' / 'p'
    effort: str = 'standard'
    tol: float = 0.0         # 2 段目のずれがこれ以下になったら、寸法を直すのをやめる（0 なら打ち切らない）
    seed: int = 0


def snap_vars(X, variables):
    """変数ごとの範囲と刻みに乗せる。"""
    X = np.atleast_2d(np.asarray(X, float))
    out = np.empty_like(X)
    for j, v in enumerate(variables):
        lo, hi = np.ceil(v.lo / v.step - 1e-9) * v.step, np.floor(v.hi / v.step + 1e-9) * v.step
        out[:, j] = np.round(np.clip(np.round(X[:, j] / v.step) * v.step, lo, hi), 6)
    return out


def cost_per_candidate(level, n_wl, n_angles, n_layers=1):
    """進み具合・所要時間の見積もりに使う、候補 1 つの相対的な重さ（次数のおよそ 2.5 乗。実測に合わせた）。"""
    return (level.norder / 49.0) ** 2.5 * n_wl * n_angles * max(1, n_layers)


def solve_candidate(task):
    """ワーカーで候補 1 つを解く → [角度, 波長] の鏡面反射率（qty='diff' なら、空気側に出る回折光 R_total − R0。2D だけ）。"""
    (kind, wl, angles_deg, norder, geo, nk_samples, env_samples, sub_samples, pol, qty) = task
    table = s4_parallel._lookup_map(wl, nk_samples)
    sub = s4_parallel._lookup_fn(wl, sub_samples)
    out = np.empty((len(angles_deg), len(wl)))
    if kind == '1d':
        from ..rcwa_s4_backend import calc_rcwa1d_s4
        for i, th in enumerate(angles_deg):
            irp, _, irs, _ = calc_rcwa1d_s4(wl, np.radians(th), geo['pitch_nm'] / 1000.0, norder, geo['layers'], table,
                                            float(np.real(env_samples[0])), sub)
            rp, rs = irp[:, norder // 2], irs[:, norder // 2]
            out[i] = {'s': rs, 'p': rp}.get(pol, 0.5 * (rs + rp))
        return out
    from ..rcwa_s4_array2d_backend import calc_array_rcwa_s4
    env = s4_parallel._lookup_fn(wl, env_samples)
    # 偏光していない光: s と p を同じ強さで重ねた光（ψ = 45°）の反射率は、入射面について対称な構造
    # （正方・六方格子の円柱や球、φ = 0）では s と p が混ざらないので、s と p の平均に等しい
    psi = {'s': 0.0, 'p': 90.0}.get(pol, 45.0)
    for i, th in enumerate(angles_deg):
        r = calc_array_rcwa_s4(wl, np.radians(th), 0.0, geo['pitch_nm'] / 1000.0, geo['pitch_nm'] / 1000.0, norder,
                               geo['layers'], table, env, sub, gamma_deg=geo.get('gamma_deg', 90.0), pol_deg=psi)
        out[i] = r['R0'] if qty != 'diff' else np.asarray(r['R_total']) - np.asarray(r['R0'])
    return out


class Context:
    """探索 1 回ぶんの条件（目標・媒質・偏光）と、計算した候補の控え。"""

    def __init__(self, family, target, nk_fns, n_env, nk_sub_fn, settings, stop_event=None, tick=None):
        self.family, self.target, self.s = family, target, settings
        self.nk_fns = {m: nk_fns[m] for m in family.materials}
        self.n_env, self.nk_sub_fn = float(np.real(n_env)), nk_sub_fn
        self.stop_event, self.tick = stop_event, tick
        self.cache = {}
        self.offset = {}      # 段階 → 細かい計算との反射率の差 [角度, 波長]（補正してずれを測る）

    def stopped(self):
        return self.stop_event is not None and self.stop_event.is_set()

    def wl(self, level):
        return np.asarray(targets.wavelengths(self.target, level.cie_stride, level.n_wl_max), float)

    def solve(self, geos, level, wl, angles, pol=None, weight=None, qty='R0'):
        """構造のリスト → [角度, 波長] の反射率のリスト（止めた分は None）。
        qty='diff' なら、空気側に出る回折光（R_total − R0）。"""
        nk = {m: s4_parallel._sample(f, wl) for m, f in self.nk_fns.items()}
        env = np.full(len(wl), complex(self.n_env))
        sub = s4_parallel._sample(self.nk_sub_fn, wl)
        if not geos:
            return []
        angles = list(angles)
        n_lay = max(len(g['layers']) for g in geos)
        conc = s4_parallel.max_concurrency(n_lay + 2, level.norder)
        # 候補がワーカーより少ないとき（2・3 段目の仕上げ、確かめる段階）は、候補 1 つを角度・波長ごとに
        # 分けて全部のワーカーで解く（分けないと、確かめる段階では 1 つのワーカーしか働かない）
        pieces = ([(i, slice(0, len(angles)), slice(0, len(wl))) for i in range(len(geos))] if len(geos) >= conc
                  else [(i, slice(a, a + 1), s) for i in range(len(geos))
                        for a, s in s4_parallel._chunks(len(angles), len(wl), -(-conc // len(geos)))])
        tasks = [(self.family.kind, wl[s], angles[a], level.norder, geos[i], {m: v[s] for m, v in nk.items()},
                  env[s], sub[s], pol or self.s.pol, qty) for i, a, s in pieces]
        size = len(angles) * len(wl)
        weights = [len(angles[a]) * len(wl[s]) / size for _, a, s in pieces]
        last = [0.0]

        def cb(done, total):
            if self.tick is not None:
                self.tick((done - last[0]) * (weight or 0.0))
            last[0] = done

        res = s4_parallel._run(solve_candidate, tasks, cb, self.stop_event, weights=weights, max_inflight=conc)
        out = [np.empty((len(angles), len(wl))) for _ in geos]
        ok = [True] * len(geos)
        for (i, a, s), r in zip(pieces, res):
            if r is None:
                ok[i] = False
            else:
                out[i][a, s] = r
        return [R if good else None for R, good in zip(out, ok)]

    def evaluate(self, X, level):
        """[候補, 変数]（project 済み）→ [(x, ずれ, 角度ごとのずれ, R), …]（計算できたものだけ）。"""
        keys = [tuple(x) for x in X]
        todo = [k for k in dict.fromkeys(keys) if (k, level) not in self.cache]
        wl = self.wl(level)
        n_lay = len(self.family.geometry(np.array(todo[0]), level)['layers']) if todo else 1
        w = cost_per_candidate(level, len(wl), len(self.target.angles), n_lay)
        res = self.solve([self.family.geometry(np.array(k), level) for k in todo], level, wl, self.target.angles, weight=w)
        for k, R in zip(todo, res):
            if R is not None and np.all(np.isfinite(R)):
                loss, per = targets.losses(self.target, wl, [r[None, :] for r in R])
                self.cache[(k, level)] = (np.array(k), float(loss[0]), per[0], R)
        if self.tick is not None and len(todo) < len(keys):
            self.tick((len(keys) - len(todo)) * w)      # 控えにあった分も進んだことにする
        return [self.cache[(k, level)] for k in dict.fromkeys(keys) if (k, level) in self.cache]

    def ranked(self, level):
        """その段階で計算した候補を、ずれの小さい順に（offset があれば、補正した反射率で測ったずれ）。"""
        out = [v for (k, l), v in self.cache.items() if l == level]
        if level in self.offset and out:
            Rs = np.array([v[3] for v in out]) + self.offset[level][None]
            loss, per = targets.losses(self.target, self.wl(level), [Rs[:, a] for a in range(Rs.shape[1])])
            out = [(v[0], float(l), p, r) for v, l, p, r in zip(out, loss, per, Rs)]
        return sorted(out, key=lambda v: v[1])

    def set_offset(self, coarse, fine, x):
        """寸法 x での、細かい段階 fine と粗い段階 coarse の反射率の差を、coarse の補正として使う。"""
        a, b = self.cache.get((tuple(x), coarse)), self.cache.get((tuple(x), fine))
        if a is None or b is None:
            return False
        self.offset[coarse] = np.array([np.interp(self.wl(coarse), self.wl(fine), rb) - ra for ra, rb in zip(a[3], b[3])])
        return True


def _spread(entries, n, variables, frac=0.03):
    """ずれの小さい順の候補から、寸法が互いに離れたものを n 個まで選ぶ。"""
    span = np.array([max(v.hi - v.lo, v.step) for v in variables])
    out = []
    for e in entries:
        if all(np.max(np.abs(e[0] - o[0]) / span) > frac for o in out):
            out.append(e)
            if len(out) >= n:
                break
    return out


def _around(ctx, level, rounds, batch, sigma0, rng, on_best, tol=0.0):
    """良い候補のまわりをくり返し探す。はじめは広く、だんだん狭く。"""
    fam = ctx.family
    V = fam.variables
    span = np.array([v.hi - v.lo for v in V])
    step = np.array([v.step for v in V])
    for r in range(rounds):
        if ctx.stopped():
            return
        elite = ctx.ranked(level)
        if not elite or (tol > 0 and elite[0][1] <= tol):
            return
        pool = _spread(elite, 5, V) or elite[:1]
        sig = np.maximum(sigma0 * 0.6 ** r * span, step)
        X = []
        for _ in range(batch):
            base = pool[min(int(rng.geometric(0.5)) - 1, len(pool) - 1)][0]
            x = base + rng.normal(size=len(V)) * sig
            if len(pool) > 2 and rng.random() < 0.4:      # 良い候補どうしの差の向きにも動かす
                a, b = rng.choice(len(pool), 2, replace=False)
                x = x + 0.5 * (pool[a][0] - pool[b][0])
            X.append(x)
        ctx.evaluate(fam.project(np.array(X)), level)
        on_best(level)


_DAMPING = (0.0, 0.03, 0.3, 3.0)
_SCALES = (1.0, 0.5)


def _polish_size(n_vars):
    return 2 * n_vars + len(_DAMPING) * len(_SCALES)


def _polish(ctx, level, iters, on_best, tol=0.0):
    """一番良い候補の寸法を 1 つずつ少し動かして色（スペクトル）の変化の傾きを測り、目標に近づく向きへ動かす
    （ガウス・ニュートン法。止まりやすさの違う歩幅をいくつか並列に試す）。良くならなければ、動かす幅を半分にする。"""
    fam = ctx.family
    V = fam.variables
    num = [j for j, v in enumerate(V) if v.unit]      # 長さの寸法だけ（ペアの数・材料の順は変えない）
    step = np.array([v.step for v in V])
    h = np.maximum(0.015 * np.array([v.hi - v.lo for v in V]), step)
    wl = ctx.wl(level)
    base = None
    for _ in range(iters):
        r = ctx.ranked(level)
        if ctx.stopped() or not r or (tol > 0 and r[0][1] <= tol):
            return
        if base is not None and np.array_equal(base, r[0][0]):
            if np.all(h[num] <= step[num]):
                return
            h = np.maximum(0.5 * h, step)
        base = r[0][0].copy()
        res0 = targets.residuals(ctx.target, wl, r[0][3])
        X = fam.project(np.array([base + s * h[j] * np.eye(len(V))[j] for j in num for s in (1, -1)]))
        off = ctx.offset.get(level, 0.0)
        got = {tuple(e[0]): e for e in ctx.evaluate(X, level)}
        J = np.zeros((res0.size, len(num)))
        for k, j in enumerate(num):
            e = [got.get(tuple(X[2 * k + i])) for i in (0, 1)]
            dx = X[2 * k][j] - X[2 * k + 1][j]
            if e[0] is not None and e[1] is not None and dx > 0:
                J[:, k] = (targets.residuals(ctx.target, wl, e[0][3] + off) - targets.residuals(ctx.target, wl, e[1][3] + off)) / dx
        JS = J * h[num]                                # 変数の大きさをそろえる
        A, g = JS.T @ JS, JS.T @ res0
        trials = []
        for lam in _DAMPING:
            d = np.linalg.lstsq(A + lam * (np.trace(A) / len(num) + 1e-12) * np.eye(len(num)), -g, rcond=1e-6)[0]
            d = d * min(1.0, 8.0 / max(np.max(np.abs(d)), 1e-12))      # 傾きが当てにできる範囲にとどめる
            for sc in _SCALES:
                x = base.copy()
                x[num] += sc * d * h[num]
                trials.append(x)
        ctx.evaluate(fam.project(np.array(trials)), level)
        on_best(level)


def plan_cost(family, target, settings):
    """探索全体の重さ（相対値）・そのうち 3 段目の重さ・段階ごとの候補数。"""
    e = _counts(family, settings.effort)
    L1, L2, L3 = family.levels
    mid = family.project(np.array([[0.5 * (v.lo + v.hi) for v in family.variables]]))[0]
    polish = _polish_size(len(family.variables))

    def one(level):
        n_wl = len(targets.wavelengths(target, level.cie_stride, level.n_wl_max))
        return cost_per_candidate(level, n_wl, len(target.angles), len(family.geometry(mid, level)['layers']))

    last = (e['verify'] + e['correct']) * one(L3) + 2 * e['correct'] * polish * one(L2)
    total = ((e['n0'] + e['r1'][0] * e['r1'][1]) * one(L1)
             + (e['top2'] + e['r2'][0] * e['r2'][1] + e['polish'] * polish) * one(L2) + last)
    return total, last, e


def plan_wall(family, target, settings):
    """探索にかかる時間の目安（相対値）と、段階ごとに同時に解ける数。
    1 回にまとめて解く候補の数と、同時に解ける数（ワーカー数と空きメモリで決まる）から、待つ時間を数える。
    候補がワーカーより少ないときは分けて解くので、およそ 重さ ÷ 同時に解ける数。"""
    import math
    e = _counts(family, settings.effort)
    L1, L2, L3 = family.levels
    mid = family.project(np.array([[0.5 * (v.lo + v.hi) for v in family.variables]]))[0]
    polish = _polish_size(len(family.variables))
    conc, one = {}, {}
    for lv in (L1, L2, L3):
        n_lay = len(family.geometry(mid, lv)['layers'])
        n_wl = len(targets.wavelengths(target, lv.cie_stride, lv.n_wl_max))
        one[lv] = cost_per_candidate(lv, n_wl, len(target.angles), n_lay)
        conc[lv] = s4_parallel.max_concurrency(n_lay + 2, lv.norder)

    def batch(lv, n, times=1):
        c = conc[lv]
        return times * one[lv] * (math.ceil(n / c) if n >= c else n / c)

    n_ver = e['verify'] + e['correct']
    wall = (batch(L1, e['n0']) + batch(L1, e['r1'][1], e['r1'][0])
            + batch(L2, e['top2']) + batch(L2, e['r2'][1], e['r2'][0]) + batch(L2, polish, e['polish'])
            + batch(L2, polish, 2 * e['correct']) + batch(L3, 1, n_ver))
    if hasattr(family, 'post'):       # 周期を動かして確かめる（中くらいの次数・0° だけ・3 つ）＋ 回折光（全部の角度・1 つ）
        wall += batch(L2, 3, n_ver) / max(1, len(target.angles)) + batch(L2, 1, n_ver)
    return wall, {'workers': s4_parallel.n_workers(), 'concurrency': min(conc.values()),
                  'mem_gb': (s4_parallel._mem_available_bytes() or 0) / 1e9}


def _counts(family, effort):
    """変数が 3 つより多い（少ない）構造では、ばらまく数と 1 回の候補数を増やす（減らす）。"""
    e = dict(EFFORTS[effort])
    f = len(family.variables) / 3.0
    e['n0'] = int(round(e['n0'] * f))
    e['r1'] = (e['r1'][0], int(round(e['r1'][1] * f)))
    e['r2'] = (e['r2'][0], max(len(family.variables) * 4, int(round(e['r2'][1] * f))))
    return e


_STAGES = ('広く探す', '寸法を直す', '確かめる')


def search(families, target, nk_fns, n_env, nk_sub_fn, settings=None, progress_cb=None, stop_event=None):
    """families（材料の組ごとの構造の種類）を順に探す → 候補のリスト（ずれの小さい順）。
    progress_cb(info): info = {'frac', 'family', 'n_families', 'label', 'stage', 'best', 'best_level'}。"""
    s = settings or Settings()
    plans = [plan_cost(f, target, s) for f in families]
    total = sum(p[0] for p in plans) or 1.0
    done_before = 0.0
    cands = []
    for i, (fam, (cost, last_cost, e)) in enumerate(zip(families, plans)):
        if stop_event is not None and stop_event.is_set():
            break
        state = {'done': 0.0, 'stage': 0, 'best': None}

        def report():
            if progress_cb is not None:
                progress_cb({'frac': min((done_before + min(state['done'], cost)) / total, 1.0), 'family': i,
                             'n_families': len(families), 'label': fam.label, 'stage': _STAGES[state['stage']],
                             'best': state['best']})

        def tick(w):
            state['done'] += w
            report()

        ctx = Context(fam, target, nk_fns, n_env, nk_sub_fn, s, stop_event, tick)

        def on_best(level):
            r = ctx.ranked(level)
            state['best'] = r[0][1] if r else None
            report()

        rng = np.random.default_rng(s.seed + 1000 * i)
        L1, L2, L3 = fam.levels
        # 1. 広く探す
        u = qmc.Sobol(len(fam.variables), seed=s.seed + i).random_base2(int(np.ceil(np.log2(e['n0']))))[:e['n0']]
        X = np.array([v.lo for v in fam.variables]) + u * np.array([v.hi - v.lo for v in fam.variables])
        ctx.evaluate(fam.project(X), L1)
        on_best(L1)
        _around(ctx, L1, *e['r1'], 0.15, rng, on_best)
        # 2. 寸法を直す
        state['stage'] = 1
        if not ctx.stopped():
            top = _spread(ctx.ranked(L1), e['top2'], fam.variables)
            ctx.evaluate(np.array([t[0] for t in top]), L2)
            on_best(L2)
            _around(ctx, L2, *e['r2'], 0.06, rng, on_best, tol=s.tol)
            _polish(ctx, L2, e['polish'], on_best, tol=s.tol)
        # 3. 確かめる（前の段階が早く終わったときは、進み具合をここで合わせる）
        state['stage'] = 2
        state['done'] = max(state['done'], cost - last_cost)
        done = []
        if not ctx.stopped():
            for x, _, _, _ in _spread(ctx.ranked(L2), e['verify'], fam.variables):
                done += [(x, L3, r) for r in ctx.evaluate(np.array([x]), L3)]
                if ctx.stopped():
                    break
            # 一番良い候補で、細かい計算と 2 段目の計算の差を測り、その差を補正して寸法を直し、もう一度確かめる
            for _ in range(e['correct']):
                best = min(done, key=lambda t: t[2][1])[0] if done else None
                if best is None or ctx.stopped() or not ctx.set_offset(L2, L3, best):
                    break
                _polish(ctx, L2, 2, lambda lv: None, tol=0.0)
                x = ctx.ranked(L2)[0][0]
                if ctx.stopped() or any(np.array_equal(x, d[0]) for d in done):
                    break
                done += [(x, L3, r) for r in ctx.evaluate(np.array([x]), L3)]
            ctx.offset.clear()
            on_best(L3)
        # 途中で止めたときは、そこまでで一番細かい計算の候補を出す
        if not done:
            for lv in (L2, L1):
                if ctx.ranked(lv):
                    done = [(r[0], lv, r) for r in _spread(ctx.ranked(lv), e['verify'], fam.variables)]
                    break
        for x, lv, (_, loss, per, R) in done:
            wl = ctx.wl(lv)
            c = {'family': fam.name, 'loss': loss, 'per_angle': [float(v) for v in per], 'angles_deg': list(target.angles),
                 'wl_nm': wl.tolist(), 'R': [r.tolist() for r in R], 'hexes': targets.hexes_of(wl, R), 'x': [float(v) for v in x],
                 'materials': list(fam.materials), 'norder': lv.norder, 'verified': lv == L3, **fam.describe(x)}
            coarse = ctx.cache.get((tuple(x), L2)) if lv == L3 else None
            if coarse is not None:
                c.update(loss_coarse=coarse[1], norder_coarse=L2.norder)
            if lv == L3 and not ctx.stopped() and hasattr(fam, 'post'):
                ctx.tick = None
                c.update(fam.post(ctx, x, L2) or {})
            cands.append(c)
        done_before += cost
    return sorted(cands, key=lambda c: c['loss'])

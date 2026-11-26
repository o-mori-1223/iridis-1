"""孤立した球（まわりは環境と同じ媒質）の Mie 理論。逆設計の結果の説明（どの大きさでどの波長に共鳴が出るか）に使う。

配列した粒子の反射率そのものは S4 で計算する（基板と隣の粒子の影響で、共鳴の波長はここの値からずれる）。
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def mie_multipoles(n_sphere, diameter_nm, wl_nm, n_env=1.0, l_max=4):
    """散乱効率 Q_sca への次数ごとの寄与 → {'a': [l, 波長]（電気）, 'b': [l, 波長]（磁気）}（l = 1 が双極子）。
    n_sphere: [波長] の複素屈折率（n + ik）。"""
    wl = np.asarray(wl_nm, float)
    m = np.asarray(n_sphere, complex) / n_env
    x = np.pi * float(diameter_nm) * n_env / wl
    mx = m * x
    out = {'a': [], 'b': []}
    for l in range(1, l_max + 1):
        jx, djx = spherical_jn(l, x), spherical_jn(l, x, True)
        yx, dyx = spherical_yn(l, x), spherical_yn(l, x, True)
        jm, djm = spherical_jn(l, mx), spherical_jn(l, mx, True)
        hx, dhx = jx + 1j * yx, djx + 1j * dyx
        psi, dpsi = x * jx, jx + x * djx
        xi, dxi = x * hx, hx + x * dhx
        psim, dpsim = mx * jm, jm + mx * djm
        a = (m * psim * dpsi - psi * dpsim) / (m * psim * dxi - xi * dpsim)
        b = (psim * dpsi - m * psi * dpsim) / (psim * dxi - m * xi * dpsim)
        f = 2.0 / x ** 2 * (2 * l + 1)
        out['a'].append(f * np.abs(a) ** 2)
        out['b'].append(f * np.abs(b) ** 2)
    return {k: np.array(v) for k, v in out.items()}


def _longest_peak(wl, v):
    i = [k for k in range(1, wl.size - 1) if v[k] > v[k - 1] and v[k] >= v[k + 1]]
    return float(wl[i[-1]]) if i else None


def dipole_resonances(nk_fn, diameter_nm, n_env=1.0, wl_lo=380.0, wl_hi=1000.0):
    """磁気双極子・電気双極子の共鳴波長 [nm]（最も長波長のピーク。範囲に無ければ None）。"""
    wl = np.arange(wl_lo, wl_hi + 0.5, 2.0)
    q = mie_multipoles(np.array([complex(nk_fn(w)) for w in wl]), diameter_nm, wl, n_env, l_max=1)
    return {'magnetic': _longest_peak(wl, q['b'][0]), 'electric': _longest_peak(wl, q['a'][0])}

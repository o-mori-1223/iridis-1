"""平らな多層膜の反射率（伝達行列法・特性行列法）。逆設計（RCWA2D/inverse/families/film.py）の高速な順計算。

横に一様な膜（列が 1 本だけの 1D 構造）なら、S4 と同じ答えを解析的に、桁違いに速く出せる
（tests/inverse_check.py で S4 と比べて確かめている）。多数の候補（膜厚の組）をまとめて計算できるよう、
膜厚は [候補, 層] の配列で受け取り、反射率を [候補, 波長] で返す。

規約: 複素屈折率は n + ik（k ≥ 0、data/nk と同じ）。時間因子 exp(-iωt)。特性行列は Born & Wolf の形
    M_j = [[cos δ, -i sin δ / η], [-i η sin δ, cos δ]],  δ = 2π N d cosθ / λ
で、層は上（入射側）から順に掛ける。η は s 偏光で N cosθ、p 偏光で N / cosθ。
"""
import numpy as np


def _normal_admittance(n, s0):
    """N cosθ（Snell の法則 n0 sinθ0 = N sinθ から）。吸収のある媒質で減衰する向き（Im ≥ 0）を選ぶ。"""
    n = np.asarray(n, complex)
    q = np.sqrt(n * n - s0 * s0 + 0j)
    flip = (q.imag < 0) | ((np.abs(q.imag) < 1e-14) & (q.real < 0))
    return np.where(flip, -q, q)


def reflectance(n_stack, d_nm, wl_nm, theta_deg, n_env, n_sub, pol):
    """多層膜の反射率。

    n_stack: [層, 波長] の複素屈折率（上＝入射側から順）。
    d_nm: [候補, 層] の膜厚 [nm]（0 の層は無いのと同じ）。
    wl_nm: [波長]。theta_deg: 入射角（環境側）。n_env: 環境の屈折率（実数）。n_sub: [波長] の基板の屈折率。
    pol: 's' または 'p'。
    戻り値: [候補, 波長] の反射率。
    """
    n_stack = np.atleast_2d(np.asarray(n_stack, complex))
    d_nm = np.atleast_2d(np.asarray(d_nm, float))
    wl = np.asarray(wl_nm, float)
    s0 = float(n_env) * np.sin(np.deg2rad(theta_deg))
    q0 = _normal_admittance(np.full(wl.shape, complex(n_env)), s0)
    qs = _normal_admittance(np.broadcast_to(np.asarray(n_sub, complex), wl.shape), s0)
    if pol == 's':
        eta0, eta_sub = q0, qs
    else:
        n_sub_ = np.broadcast_to(np.asarray(n_sub, complex), wl.shape)
        eta0, eta_sub = complex(n_env) ** 2 / q0, n_sub_ * n_sub_ / qs
    n_cand = d_nm.shape[0]
    B = np.ones((n_cand, wl.size), complex)
    C = np.broadcast_to(eta_sub, (n_cand, wl.size)).astype(complex)
    for j in range(n_stack.shape[0] - 1, -1, -1):   # 基板側の層から順に掛ける
        q = _normal_admittance(n_stack[j], s0)
        eta = q if pol == 's' else n_stack[j] * n_stack[j] / q
        delta = 2.0 * np.pi * d_nm[:, j:j + 1] * q[None, :] / wl[None, :]
        c, s = np.cos(delta), np.sin(delta)
        B, C = c * B - 1j * s / eta * C, -1j * eta * s * B + c * C
    r = (eta0 * B - C) / (eta0 * B + C)
    return np.abs(r) ** 2


def reflectance_unpolarized(n_stack, d_nm, wl_nm, theta_deg, n_env, n_sub, pol='avg'):
    """pol = 's' / 'p' / 'avg'（s と p の平均 = 偏光していない光）。"""
    if pol in ('s', 'p'):
        return reflectance(n_stack, d_nm, wl_nm, theta_deg, n_env, n_sub, pol)
    return 0.5 * (reflectance(n_stack, d_nm, wl_nm, theta_deg, n_env, n_sub, 's') +
                  reflectance(n_stack, d_nm, wl_nm, theta_deg, n_env, n_sub, 'p'))

"""third_party/S4（本家 victorliu/S4、lab/S4 からコピーしたソース）を ctypes で呼ぶ
Pythonラッパー。`import RCWA2D.s4lab as S4` として、従来の `import S4`
（WSLの ~/s4env に入っている phoebe-p/S4 フォークのPython拡張）の置き換えに使う。

共有ライブラリ `RCWA2D/libs4lab.so` は `bash third_party/S4/build_s4lab.sh`（WSL）で
ビルドする。本家のPythonバインディング(main_python.c)はコアAPIに追従しておらず
ビルドできないため、`third_party/S4/s4lab_shim.c` 経由で S4.h のAPIを直接呼んでいる。

提供するのは rcwa_s4_backend.py / rcwa_s4_array2d_backend.py が実際に使うメソッドだけで、
名前・キーワード引数・戻り値の形は従来の S4 Python拡張と同じ:
    S = New(Lattice=period_um or ((ux,uy),(vx,vy)), NumBasis=n)
    S.SetMaterial(Name, Epsilon)
    S.AddLayer(Name, Thickness, Material)
    S.AddLayerCopy(Name, Thickness, Layer)                             # 同じ形の層の固有モードを使い回す
    S.SetRegionRectangle(Layer, Material, Center, Angle, Halfwidths)   # Angle は度
    S.SetRegionCircle(Layer, Material, Center, Radius)
    S.SetOptions(PolarizationDecomposition=bool, Verbosity=int)
    S.SetFrequency(f)                                                  # f = 1/波長
    S.SetExcitationPlanewave(IncidenceAngles=(theta,phi), sAmplitude, pAmplitude)  # 度
    S.GetPowerFluxByOrder(Layer, zOffset) -> ((fwd, bwd), ...)         # 各 complex
    S.GetBasisSet() -> ((g1, g2), ...)

1D格子の矩形は本家コアの規約に従い区間(interval)として登録される（Halfwidths の
第2成分は無視される）。
"""
import ctypes
import math
import os

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs4lab.so')

try:
    _lib = ctypes.CDLL(_LIB_PATH)
except OSError as e:
    # Windows や未ビルド環境。呼び出し側の `except ImportError` で HAVE_S4=False になる。
    raise ImportError(f"S4 共有ライブラリを読み込めません: {_LIB_PATH} ({e})。"
                      "WSLで `bash third_party/S4/build_s4lab.sh` を実行してください。") from e

# libs4lab.so がリンクしている OpenBLAS を1スレッドに固定する。S4が扱う行列（数十〜
# 数百次）ではBLASのスレッド並列は逆効果で、実測で1スレッドの方が1.6〜1.7倍速い
# （並列化は RCWA2D/s4_parallel.py のプロセス並列で行う）。
# libs4lab.so は libblas.so.3 にリンクしている。Ubuntu で libopenblas0 を入れると update-alternatives で
# libblas.so.3 自体が OpenBLAS になるので、まずそれ（S4 が実際に使う方）を探す。参照実装の BLAS
# なら openblas_set_num_threads が無く、スレッド数は変えられない（常に1スレッド）。
_openblas = None
for _blas in ('libblas.so.3', 'libopenblas.so.0', 'libopenblas.so'):
    try:
        _openblas = ctypes.CDLL(_blas)
        _openblas.openblas_set_num_threads(1)
        break
    except (OSError, AttributeError):
        _openblas = None


def set_blas_threads(n):
    """OpenBLAS のスレッド数を変える（このプロセスだけ）。NumBasis が大きくメモリの都合で同時に
    解く数が絞られたとき、空いたコアを行列計算に回すために s4_parallel が使う。"""
    if _openblas is not None:
        _openblas.openblas_set_num_threads(max(1, int(n)))

_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_int_p = ctypes.POINTER(ctypes.c_int)
_d, _i, _s, _p = ctypes.c_double, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p

_lib.s4l_new.restype = _p
_lib.s4l_new.argtypes = [_c_double_p, _i]
_lib.s4l_destroy.restype = None
_lib.s4l_destroy.argtypes = [_p]
_lib.s4l_get_basis.restype = _i
_lib.s4l_get_basis.argtypes = [_p, _c_int_p]
_lib.s4l_set_basis.restype = _i
_lib.s4l_set_basis.argtypes = [_p, _i, _c_int_p]
_lib.s4l_set_material.argtypes = [_p, _s, _d, _d]
_lib.s4l_add_layer.argtypes = [_p, _s, _d, _s]
_lib.s4l_add_layer_copy.argtypes = [_p, _s, _d, _s]
_lib.s4l_set_region_rectangle.argtypes = [_p, _s, _s, _d, _d, _d, _d, _d]
_lib.s4l_set_region_circle.argtypes = [_p, _s, _s, _d, _d, _d]
_lib.s4l_set_region_polygon.argtypes = [_p, _s, _s, _d, _d, _i, _c_double_p]
_lib.s4l_set_frequency.argtypes = [_p, _d, _d]
_lib.s4l_set_excitation_planewave.argtypes = [_p, _d, _d, _d, _d, _d, _d]
_lib.s4l_get_power_flux_by_order.argtypes = [_p, _s, _d, _c_double_p]
_lib.s4l_set_polarization_decomposition.argtypes = [_p, _i]
_lib.s4l_set_verbosity.argtypes = [_p, _i]
_lib.s4l_set_polarization_options.argtypes = [_p, _i, _i, _i]
for _f in ('s4l_set_material', 's4l_add_layer', 's4l_add_layer_copy', 's4l_set_region_rectangle', 's4l_set_region_circle',
           's4l_set_region_polygon',
           's4l_set_frequency', 's4l_set_excitation_planewave', 's4l_get_power_flux_by_order',
           's4l_set_polarization_decomposition', 's4l_set_verbosity', 's4l_set_polarization_options'):
    getattr(_lib, _f).restype = _i

# S4_internal.h の Solution error codes
_ERRORS = {
    1: 'allocation error', 9: 'lattice n_G not set', 10: 'copy referenced an unknown layer',
    11: 'a copy of a copy was made', 12: 'duplicate layer name', 13: 'layer not found',
    14: 'no layers', 15: 'material not found', 16: 'invalid 1D layer patterning',
}


_RAYLEIGH_NUDGE = 1e-9   # GetPowerFluxByOrder のNaN回避用の相対周波数シフト


def _select_basis_2d(Lr, nG):
    """2D格子の基底を円形打ち切り（|G|の小さい順にnG個）で選び、最後の殻（同じ|G|の組）が
    途中で切れる場合はその殻を丸ごと除く——本家 gsel.c の Gsel_circular と同じ方針だが、
    同一殻の判定を相対許容誤差1e-9で行う。gsel.c の判定（|G|^2 の差を 2*DBL_EPSILON*|G|
    と比較）は六方格子などで丸め誤差に負けて殻が中途半端に残り、反転対称でない基底
    （例: NumBasis=49 で (-3,-4) はあるのに (3,4) が無い48本）になってしまうため。
    """
    ux, uy, vx, vy = Lr
    d = 1.0 / (ux * vy - uy * vx)
    k1 = (d * vy, -d * vx)          # S4_Lattice_Reciprocate と同じ（2π を除いた逆格子）
    k2 = (-d * uy, d * ux)
    u2 = k1[0] ** 2 + k1[1] ** 2
    v2 = k2[0] ** 2 + k2[1] ** 2
    uv = k1[0] * k2[0] + k1[1] * k2[1]
    sin_t = math.sqrt(max(1.0 - uv * uv / (u2 * v2), 1e-12))
    radius = math.sqrt(nG * abs(k1[0] * k2[1] - k1[1] * k2[0]) / math.pi) + math.sqrt(u2) + math.sqrt(v2)
    ue = 1 + int(radius / (math.sqrt(u2) * sin_t))
    ve = 1 + int(radius / (math.sqrt(v2) * sin_t))
    cand = sorted(((i * i * u2 + 2 * i * j * uv + j * j * v2, i, j)
                   for i in range(-ue, ue + 1) for j in range(-ve, ve + 1)))
    n = min(nG, len(cand))
    if n < len(cand):
        tol = 1e-9 * cand[n][0]
        while n > 1 and abs(cand[n][0] - cand[n - 1][0]) <= tol:
            n -= 1
    return [(i, j) for _, i, j in cand[:n]]


def _check(fname, ret):
    if ret != 0:
        msg = _ERRORS.get(ret, 'invalid argument' if ret < 0 else 'unknown error')
        raise RuntimeError(f"{fname}: S4 error {ret} ({msg})")


def _b(s):
    return s.encode('utf-8')


class Simulation:
    def __init__(self, Lattice, NumBasis):
        if isinstance(Lattice, (int, float)):
            Lr = (float(Lattice), 0.0, 0.0, 0.0)
        else:
            (ux, uy), (vx, vy) = Lattice
            Lr = (float(ux), float(uy), float(vx), float(vy))
        self._S = _lib.s4l_new((ctypes.c_double * 4)(*Lr), int(NumBasis))
        if not self._S:
            raise RuntimeError("New: S4_Simulation_New failed")
        if Lr[2] != 0.0 or Lr[3] != 0.0:
            basis = _select_basis_2d(Lr, int(NumBasis))
            G = (ctypes.c_int * (2 * len(basis)))(*[g for ij in basis for g in ij])
            _check('New', _lib.s4l_set_basis(self._S, len(basis), G))
        self._freq = 1.0 + 0j
        self._nudged = False

    def __del__(self):
        S, self._S = getattr(self, '_S', None), None
        if S:
            _lib.s4l_destroy(S)

    def SetMaterial(self, Name, Epsilon):
        eps = complex(Epsilon)
        _check('SetMaterial', _lib.s4l_set_material(self._S, _b(Name), eps.real, eps.imag))

    def AddLayer(self, Name, Thickness, Material):
        _check('AddLayer', _lib.s4l_add_layer(self._S, _b(Name), float(Thickness), _b(Material)))

    def AddLayerCopy(self, Name, Thickness, Layer):
        """既存の層 Layer と同じ形・材質の層を足す（厚さだけ変えられる）。固有モードは Layer の分を
        使い回すので、同じ形の層が何枚あっても固有値計算は 1 回で済む。コピーに形を足さないこと。"""
        _check('AddLayerCopy', _lib.s4l_add_layer_copy(self._S, _b(Name), float(Thickness), _b(Layer)))

    def SetRegionRectangle(self, Layer, Material, Center, Angle, Halfwidths):
        _check('SetRegionRectangle', _lib.s4l_set_region_rectangle(
            self._S, _b(Layer), _b(Material), float(Center[0]), float(Center[1]),
            float(Angle), float(Halfwidths[0]), float(Halfwidths[1])))

    def SetRegionCircle(self, Layer, Material, Center, Radius):
        _check('SetRegionCircle', _lib.s4l_set_region_circle(
            self._S, _b(Layer), _b(Material), float(Center[0]), float(Center[1]), float(Radius)))

    _BASES = {'Default': 0, 'Jones': 1, 'Normal': 2}

    def SetRegionPolygon(self, Layer, Material, Center, Angle, Vertices):
        """Vertices: Center からの相対座標 (x, y) のリスト。時計回りなら反時計回りに直して渡す。
        Angle（度）は 0 のみ対応。"""
        if Angle:
            raise NotImplementedError("SetRegionPolygon: Angle != 0 は未対応")
        v = [(float(x), float(y)) for x, y in Vertices]
        area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(v, v[1:] + v[:1]))
        if area2 < 0:
            v = v[::-1]
        flat = (ctypes.c_double * (2 * len(v)))(*[c for xy in v for c in xy])
        _check('SetRegionPolygon', _lib.s4l_set_region_polygon(
            self._S, _b(Layer), _b(Material), float(Center[0]), float(Center[1]), len(v), flat))

    def SetOptions(self, PolarizationDecomposition=None, Verbosity=None, DiscretizationResolution=None,
                   PolarizationBasis=None, ConserveMemory=None, **kwargs):
        """従来の S4 拡張と同名のオプション。PolarizationBasis は 'Default' / 'Jones' / 'Normal'。"""
        if kwargs:
            raise NotImplementedError(f"SetOptions: 未対応のオプション {sorted(kwargs)}")
        if PolarizationDecomposition is not None:
            _check('SetOptions', _lib.s4l_set_polarization_decomposition(
                self._S, 1 if PolarizationDecomposition else 0))
        if Verbosity is not None:
            _check('SetOptions', _lib.s4l_set_verbosity(self._S, int(Verbosity)))
        if DiscretizationResolution is not None or PolarizationBasis is not None or ConserveMemory is not None:
            _check('SetOptions', _lib.s4l_set_polarization_options(
                self._S,
                -1 if DiscretizationResolution is None else int(DiscretizationResolution),
                -1 if PolarizationBasis is None else self._BASES[PolarizationBasis],
                -1 if ConserveMemory is None else int(bool(ConserveMemory))))

    def SetFrequency(self, f):
        f = complex(f)
        _check('SetFrequency', _lib.s4l_set_frequency(self._S, f.real, f.imag))
        self._freq = f
        self._nudged = False

    def SetExcitationPlanewave(self, IncidenceAngles, sAmplitude=0.0, pAmplitude=0.0, Order=0):
        if Order != 0:
            raise NotImplementedError("SetExcitationPlanewave: Order != 0 は未対応")
        theta, phi = IncidenceAngles
        s, p = complex(sAmplitude), complex(pAmplitude)
        _check('SetExcitationPlanewave', _lib.s4l_set_excitation_planewave(
            self._S, float(theta), float(phi), s.real, s.imag, p.real, p.imag))

    def GetBasisSet(self):
        n = _lib.s4l_get_basis(self._S, None)
        G = (ctypes.c_int * (2 * n))()
        _lib.s4l_get_basis(self._S, G)
        return tuple((G[2*i], G[2*i + 1]) for i in range(n))

    def _power_flux(self, Layer, zOffset):
        n = _lib.s4l_get_basis(self._S, None)
        P = (ctypes.c_double * (4 * n))()
        _check('GetPowerFluxByOrder', _lib.s4l_get_power_flux_by_order(
            self._S, _b(Layer), float(zOffset), P))
        # 解の初期化(Simulation_InitSolution)で n_G が減ることがあるので取り直す（増えることはない）
        n = _lib.s4l_get_basis(self._S, None)
        return P, n

    def GetPowerFluxByOrder(self, Layer, zOffset=0.0):
        P, n = self._power_flux(Layer, zOffset)
        if not self._nudged and any(math.isnan(P[i]) for i in range(4 * n)):
            # 回折次数がちょうどグレージング(kz=0、レイリー異常)になる波長では、本家コアの
            # rcwa.cpp が q=0 で割ってNaNになる。異常点は測度0の特異点で物理量は連続なので、
            # 周波数を相対1e-9だけずらして解き直し、極限値を返す（実測: 3000nmピッチ・
            # 垂直入射の λ=500/600nm で、ずらした値は ±1e-6nm 両側の値と一致）。
            # ずらした状態は次の SetFrequency() まで維持し、同じ波長点の他の層の
            # GetPowerFluxByOrder() も同じ解から取る（上下の層で値の整合を保つため）。
            f = self._freq * (1.0 + _RAYLEIGH_NUDGE)
            _check('SetFrequency', _lib.s4l_set_frequency(self._S, f.real, f.imag))
            self._nudged = True
            P, n = self._power_flux(Layer, zOffset)
        return tuple((complex(P[4*i], P[4*i + 2]), complex(P[4*i + 1], P[4*i + 3]))
                     for i in range(n))


def New(Lattice, NumBasis):
    return Simulation(Lattice=Lattice, NumBasis=NumBasis)

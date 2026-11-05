"""ナノ円盤（またはその他の円形断面ナノ構造）の**2次元周期アレイ**をS4で解く
バックエンド。

## なぜこれが必要か
`rcwa_s4_backend.py`（1D backend）は`S4.New(Lattice=period_um, ...)`と
スカラーのLatticeを渡しており、これはS4内部では1次元格子
（x方向だけに周期、y方向は無限一様）として扱われる——koumyouアプリの
「列（column）」構造、すなわち1D grating専用。

一方、修士研究計画（配置パターン・配置間隔・観察角の変化が発色特性に与える
影響の評価）は本質的に**2次元周期アレイ**（x,y両方向に周期的に並んだ
ナノ粒子）の問題であり、1D backendでは原理的に扱えない
（列＝x方向だけの周期構造なので、配置間隔を「x方向とy方向で別々に」
振ったり、正方格子/六方格子といった「配置パターン」を変えたりすることが
そもそも表現できない）。

MEEP3D（`MEEP3D/`）は逆に**周期性が一切ない孤立1粒子**を意図的に選んだ
設計（HANDOVER.md参照）なので、これも配置間隔・パターンの効果を出せない
（隣の粒子がそもそも存在しない系）。

このモジュールは、S4が本来サポートしている2次元格子
（`Lattice=((ux,uy),(vx,vy))`）と円形領域パターニング
（`SetRegionCircle(layer, material, center, radius)`）を使い、
「基板上に周期的に並んだナノ円盤アレイ」を実際に周波数領域RCWAで解く。
ナノ円盤はz方向に断面が変化しない円柱形状なので、**staircase近似なしに
厳密に**表現できる（MEEP3Dのように孤立粒子をFDTDで解く場合と違い、
球のようにz方向の断面が変わる形状は複数の円形領域を層状に積んで近似する
必要があるが、円盤単体なら1層で済む）。

## S4のPython API仕様（実際にS4/main_python.cを直接読んで確認したもの、
推測ではない）
    S = S4.New(Lattice=((ux_um, uy_um), (vx_um, vy_um)), NumBasis=norder)
        # 2Dの場合はタプルのタプル。1D backendのようなスカラーではない
        # （main_python.c の lattice_converter() で分岐しているのを確認済み）。
    S.SetRegionCircle(Layer=name, Material=mat, Center=(cx_um, cy_um), Radius=r_um)
        # 円形領域。楕円が要る場合はSetRegionEllipse(...,Angle,Halfwidths)も
        # 同様に存在する（未使用）。
    S.GetBasisSet() -> ((g1,g2), (g1,g2), ...)
        # 2D格子では(g1,g2)が両方とも意味を持つ（1D backendのdocstringでは
        # 常にg2==0と書いていたが、それは1D格子限定の話）。
    S.GetPowerFluxByOrder(Layer=name, zOffset=0.0) -> ((fwd,bwd), (fwd,bwd), ...)
        # GetBasisSet()と同じ並び順。1D backendと同じ符号規約
        # （backward.realの符号を反転したものがR、forward.realがそのままT、
        # どちらも0次光の入射パワーで規格化）とHave同じ意味のはずだが、
        # 2D格子で改めて検証（下記の__main__セルフテスト参照）。
    その他（SetMaterial, SetExcitationPlanewave, SetFrequency,
    PolarizationDecomposition等）は1D backendと同一。

## 検証状況
`__main__`セルフテストで以下を確認する:
1. 円盤半径を実質0（数値的に安定な最小値）にしたとき、任意のピッチ・
   格子形状によらず、単純なFresnel反射率（基板と同じ）に一致すること
   （周期構造としての効果が消える極限）。
2. 無損失材質（TiO2, SiO2、微小損失込み）でエネルギー保存則
   R_total + T_total ≈ 1 が成り立つこと。
3. 環境（超格子側媒質）を「真空(n=1固定)」から「大気(data/nk/Air.nk、
   波長ごとの表引き)」に切り替えても同じ結果になること（後述の
   `nk_env_fn`一般化の検証）。

## 環境(env)側の屈折率は基板・薄膜と同じ「波長依存関数」で受け取る
【2026-09-25追記】初版では`n_env_val`という単純なスカラーで環境の屈折率を
受け取っていたが、これは`nk_subst_fn`/`nk_fn_map`（薄膜材質・基板は常に
`wl -> complex`の関数）と扱いが不統一だった。koumyouアプリ側の
`data/nk/Air.nk`（n=1.00, k=0.00、全波長で定数——「真空」と数値上は
同じだが独立した名前付き材質として選べる）のような、名前付きnkテーブルを
環境側にもそのまま使えるようにするため、`n_env_val`を`nk_env_fn`
（`wl_nm -> complex`の関数）に変更した。呼び出し側は
`lambda wl: 1.0+0j`（真空、分散なし）でも、既存の`make_nk_fn(['Air'])[0]`
（`data/nk/Air.nk`を表引き）でも、任意のカスタムn+ikでも、同じ引数として
渡せる——薄膜・基板の材質指定と完全に同じ規約になった。

## 角度掃引・球体対応【2026-09-25追記】
- `calc_array_rcwa_s4_angle_sweep()`: 入射角θの配列を受け取り、S4のsim/
  ジオメトリ構築を1回だけ行って角度×波長の全点をループする角度掃引版。
  1点ずつの計算処理は`calc_array_rcwa_s4()`と共通の`_solve_one_point()`
  ヘルパーに切り出して重複を避けている（この共通化のリファクタリングは
  既存の`__main__`セルフテストCheck 1-4を再実行して結果が変わっていない
  ことを確認済み）。
- `sphere_disk_layers()`: 球を複数の薄い円盤（staircase近似）に分解した
  `layers`を返すヘルパー。円盤はz方向に断面が変化しない円柱なので1層で
  厳密に表現できるが、球は断面が変化するため近似が必要——n_slicesを
  増やすほど真の球に収束する（収束確認は`__main__`のCheck 6参照）。
  ジオメトリを組むだけのヘルパーなので、`calc_array_rcwa_s4()`/
  `calc_array_rcwa_s4_angle_sweep()`本体には一切変更が要らない
  （`layers`の内容が円盤1層か球のstaircase近似かの違いだけ）。

## 六方格子対応【2026-09-30追記】
- これまで`calc_array_rcwa_s4()`/`calc_array_rcwa_s4_angle_sweep()`は
  `Lattice=((pitch_x_um,0),(0,pitch_y_um))`と直交格子ベクトルを決め打ち
  していたため、正方・矩形格子しか表現できなかった（本ファイル冒頭の
  「配置パターン・配置間隔」という研究目的からすると、六方格子（円盤/球の
  六方最密配置）が表現できないのはそもそもの目的に対して片手落ちだった）。
- `lattice_vectors_um(pitch_x_um, pitch_y_um, gamma_deg)`を新設し、両関数に
  `gamma_deg=90.0`（デフォルト、既存呼び出しと完全互換）を追加した。
  a1=(pitch_x_um,0), a2=(pitch_y_um*cos(gamma_deg), pitch_y_um*sin(gamma_deg))
  という一般の格子ベクトル対を組み立てる——`pitch_x_um==pitch_y_um==a`かつ
  `gamma_deg=60.0`とすると、a1・a2に加えa1-a2も長さaになり、単位胞あたり
  粒子1個の六方最密配置（各粒子の最近接粒子6個が全て距離a）になる。
  粒子自体は常に単位胞原点に1個だけ置く設計（`build_array_geometry`は
  無変更）——六方格子は「配置」を格子ベクトルの向きだけで表現でき、
  基底（1胞あたりの粒子数）を増やす必要がないため。
- 検証は`__main__`のCheck 7(エネルギー保存則)・Check 8(六方格子のC6対称性:
  φ=0°とφ=60°で入射条件を60°回転させても格子の見え方は変わらないはずなので
  R0/R_totalが一致することを確認)を参照。
"""
import numpy as np

# 2026-10-26: S4本体を third_party/S4（本家 victorliu/S4 のソース）に置き換え、
# RCWA2D/s4lab.py（ctypes ラッパー、従来の S4 Python拡張と同じメソッド名・引数）経由で呼ぶ。
# 2D格子の基底は s4lab 側で反転対称な完全殻として選び直している（s4lab._select_basis_2d）。
try:
    try:
        from RCWA2D import s4lab as S4
    except ModuleNotFoundError:   # RCWA2D/ 内のスクリプトとして直接実行した場合
        import s4lab as S4
    HAVE_S4 = True
except ImportError:
    S4 = None
    HAVE_S4 = False

ENV_MATERIAL = '__ENV__'
SUBST_MATERIAL = '__SUBSTRATE__'


def _require_s4():
    if not HAVE_S4:
        raise RuntimeError("S4 (RCWA2D/libs4lab.so) を読み込めません。WSLで "
                           "`bash third_party/S4/build_s4lab.sh` を実行してビルドしてください。")


def lattice_vectors_um(pitch_x_um, pitch_y_um, gamma_deg=90.0):
    """S4の`Lattice=((ux,uy),(vx,vy))`用に、2つの格子ベクトルを組み立てる。

    a1 = (pitch_x_um, 0)、a2 = (pitch_y_um*cos(gamma), pitch_y_um*sin(gamma))。
    gamma_deg は a1-a2 間の角度で、90.0(デフォルト)なら正方/矩形格子
    （pitch_x_um==pitch_y_umなら正方、異なれば矩形）。**六方格子**は
    pitch_x_um == pitch_y_um == a（格子定数）かつ gamma_deg=60.0 で得られる
    ——このときa1,a2に加えa1-a2も長さaになり、1粒子/単位胞の六方最密配置
    （各粒子が6個の最近接粒子を持つ）を表す。gamma_deg=90.0は浮動小数点の
    cos(90°)≈6.1e-17によるノイズを避けるため厳密に0.0として扱う（既存の
    正方/矩形格子の結果に対する後方互換性のため）。
    """
    a1 = (float(pitch_x_um), 0.0)
    if gamma_deg == 90.0:
        a2 = (0.0, float(pitch_y_um))
    else:
        gamma_rad = np.radians(gamma_deg)
        a2 = (pitch_y_um * np.cos(gamma_rad), pitch_y_um * np.sin(gamma_rad))
    return a1, a2


def build_array_geometry(sim, layers, material_names):
    """`layers`（top-first、各要素は
    {'thickness_nm': float, 'bg_material': str,
     'disks': [{'material': str, 'cx_nm': float, 'cy_nm': float, 'radius_nm': float}, ...],
     'rects': [{'material': str, 'cx_nm': float, 'cy_nm': float,
                'hw_x_nm': float, 'hw_y_nm': float}, ...]}）
    から、S4のレイヤースタック＋円形/矩形パターンを構築する（材質は仮のeps=1で
    登録、実際の値は波長ごとにupdate_materials()で設定）。'rects'は
    structure_builder.regions_to_layers_2d()が「段付き(コンフォーマル)多層膜」
    の基板上げ領域を表現するために使う——`S.SetRegionRectangle(Center=,
    Angle=0.0, Halfwidths=(hw_x,hw_y))`は1D backend(rcwa_s4_backend.py)で
    実機検証済みのAPI呼び出しと同じで、ここではHalfwidthsの両成分を有限値に
    することで真の2D矩形領域として使う（1D側はHalfwidths=(hw,big_y)で
    疑似無限化していただけで、API自体は2D格子でも同様に使えるはず——
    __main__のCheck 9で実際に検証する）。
    """
    _require_s4()

    sim.SetMaterial(Name=ENV_MATERIAL, Epsilon=1.0 + 0j)
    sim.SetMaterial(Name=SUBST_MATERIAL, Epsilon=1.0 + 0j)
    for name in sorted(material_names):
        sim.SetMaterial(Name=name, Epsilon=1.0 + 0j)

    sim.SetOptions(PolarizationDecomposition=True)

    sim.AddLayer(Name='__env_super__', Thickness=0.0, Material=ENV_MATERIAL)

    for i, L in enumerate(layers):
        lname = f'L{i}'
        thickness_um = L['thickness_nm'] / 1000.0
        sim.AddLayer(Name=lname, Thickness=thickness_um, Material=L['bg_material'])
        for d in L.get('disks', []):
            if d['radius_nm'] <= 0:
                continue
            sim.SetRegionCircle(
                Layer=lname, Material=d['material'],
                Center=(d['cx_nm'] / 1000.0, d['cy_nm'] / 1000.0),
                Radius=d['radius_nm'] / 1000.0,
            )
        for r in L.get('rects', []):
            if r['hw_x_nm'] <= 0 or r['hw_y_nm'] <= 0:
                continue
            sim.SetRegionRectangle(
                Layer=lname, Material=r['material'],
                Center=(r['cx_nm'] / 1000.0, r['cy_nm'] / 1000.0), Angle=0.0,
                Halfwidths=(r['hw_x_nm'] / 1000.0, r['hw_y_nm'] / 1000.0),
            )
        for pg in L.get('polygons', []):   # 'vertices_nm' は絶対座標 (x, y) のリスト
            sim.SetRegionPolygon(
                Layer=lname, Material=pg['material'], Center=(0.0, 0.0), Angle=0.0,
                Vertices=[(x / 1000.0, y / 1000.0) for x, y in pg['vertices_nm']],
            )

    sim.AddLayer(Name='__subst__', Thickness=0.0, Material=SUBST_MATERIAL)


def simplify_full_cell_regions(layers, pitch_x_um, pitch_y_um, gamma_deg=90.0):
    """単位胞全体を覆う矩形/円（例: ピッチ300x900nmに300x900nmの基板上げ矩形）を、
    その層の背景材質に置き換えた layers を返す（元の layers は変更しない）。

    セル内に材質境界が無いと、PolarizationDecomposition（Liの因子分解用の法線ベクトル場）が
    定義できず、S4 の層の固有値問題が失敗して全点 NaN になる（実測、2026-10-26）。
    置き換えは厳密に等価（偏光分解を切って解いた値と全桁一致を確認）。
    矩形格子（gamma_deg=90）のみ判定する。それ以外の格子で同様の問題が起きた場合は
    _solve_one_point の偏光分解なしへのフォールバックが拾う。"""
    if gamma_deg != 90.0:
        return layers
    px, py = pitch_x_um * 1000.0, pitch_y_um * 1000.0
    eps = 1e-9 * max(px, py)

    def covers(shape, is_rect):
        if is_rect:
            return 2 * shape['hw_x_nm'] >= px - eps and 2 * shape['hw_y_nm'] >= py - eps
        return shape['radius_nm'] >= 0.5 * np.hypot(px, py) - eps

    out = []
    for L in layers:
        # build_array_geometry の描画順（disks → rects）。後に描いた領域が前の領域を上書きする。
        shapes = [(d, False) for d in L.get('disks', []) if d['radius_nm'] > 0] + \
                 [(r, True) for r in L.get('rects', []) if r['hw_x_nm'] > 0 and r['hw_y_nm'] > 0]
        last_full = max((i for i, (s, is_rect) in enumerate(shapes) if covers(s, is_rect)), default=None)
        if last_full is not None and not L.get('polygons'):   # 多角形を含む層は描画順の扱いが別なので触らない
            # 最後の全面領域の材質を層の背景にし、それより前の領域（全て隠れる）は捨て、
            # 後に描かれる領域（その上に乗る）は順序どおり残す。
            rest = shapes[last_full + 1:]
            L = {'thickness_nm': L['thickness_nm'], 'bg_material': shapes[last_full][0]['material'],
                 'disks': [s for s, is_rect in rest if not is_rect],
                 'rects': [s for s, is_rect in rest if is_rect]}
        out.append(L)
    return out


def reduce_uniform_period(layers, pitch_x_um, pitch_y_um, gamma_deg, norder):
    """構造全体が x（または y）方向に一様なら、その方向の周期を十分小さくした等価な
    (layers, pitch_x_um, pitch_y_um) を返す。一様でなければそのまま返す。

    一様な方向の回折次数は構造と結合しない（その方向の波数が保存される）ので、周期を
    変えても物理は同じ。一方 S4 は NumBasis 個の次数を円形に (m, n) へ割り振るため、
    一様な方向にも次数を使い、それと直交する方向の次数が足りなくなる。実測（横倒しの
    無限に長い棒、TiO2 r=100nm、周期532nm、650nm、E が棒に垂直）: R0 の真値 ≈0.054 に対し
    NumBasis 41/81/161/321 で 0.015/0.027/0.037/0.043 と収束が非常に遅い。周期を小さくして
    全次数を棒に垂直な方向に回すと 0.051/0.053/0.054/0.054。
    新しい周期は、その方向の最初の次数が直交方向の最大の次数より外側（基底に入らない）になる
    大きさ（直交方向の周期 / (2·NumBasis)）。矩形格子（gamma_deg=90）で、円・多角形がなく、
    すべての矩形がその方向の周期いっぱいに広がっている場合だけ（平坦膜のみの構造は対象外）。"""
    if abs(gamma_deg - 90.0) > 1e-9:
        return layers, pitch_x_um, pitch_y_um
    rects = [r for L in layers for r in L.get('rects', []) if r['hw_x_nm'] > 0 and r['hw_y_nm'] > 0]
    if not rects or any(L.get('disks') or L.get('polygons') for L in layers):
        return layers, pitch_x_um, pitch_y_um
    px, py = pitch_x_um * 1000.0, pitch_y_um * 1000.0
    for axis, period, other in (('x', px, py), ('y', py, px)):
        key = 'hw_x_nm' if axis == 'x' else 'hw_y_nm'
        if all(2 * r[key] >= period * (1 - 1e-9) for r in rects):
            new = other / (2.0 * max(int(norder), 1))
            out = [dict(L, rects=[dict(r, **{key: new / 2, ('cx_nm' if axis == 'x' else 'cy_nm'): 0.0})
                                  for r in L.get('rects', [])]) for L in layers]
            return (out, new / 1000.0, pitch_y_um) if axis == 'x' else (out, pitch_x_um, new / 1000.0)
    return layers, pitch_x_um, pitch_y_um


def update_materials(sim, wl_nm, material_names, nk_fn_map, nk_env_fn, nk_subst_fn):
    n_env = complex(nk_env_fn(wl_nm))
    sim.SetMaterial(Name=ENV_MATERIAL, Epsilon=n_env * n_env)
    n_subst = complex(nk_subst_fn(wl_nm))
    sim.SetMaterial(Name=SUBST_MATERIAL, Epsilon=n_subst * n_subst)
    for name in material_names:
        n = complex(nk_fn_map[name](wl_nm))
        sim.SetMaterial(Name=name, Epsilon=n * n)


def _collect_material_names(layers):
    names = set()
    for L in layers:
        names.add(L['bg_material'])
        for d in L.get('disks', []):
            names.add(d['material'])
        for r in L.get('rects', []):
            names.add(r['material'])
        for pg in L.get('polygons', []):
            names.add(pg['material'])
    names.discard(ENV_MATERIAL)
    names.discard(SUBST_MATERIAL)
    return sorted(names)


class _BasisCache:
    """(g1,g2)==(0,0)の行番号は、sim構築後の初回Simulateで初めて解決できる
    （GetBasisSet()内部でSimulation_InitSolution()が呼ばれる）。角度・波長を
    ループする間は同じsimオブジェクトを使い続けるので、この行番号自体は
    ループ全体で不変——1回だけ解決してキャッシュする。"""
    def __init__(self):
        self.basis = None
        self.idx0 = None

    def resolve(self, sim):
        if self.basis is None:
            self.basis = sim.GetBasisSet()
            self.idx0 = next(i for i, (g1, g2) in enumerate(self.basis) if g1 == 0 and g2 == 0)
        return self.basis, self.idx0


def _solve_one_point(sim, wl_nm, theta_rad, phi_rad, material_names, nk_fn_map,
                      nk_env_fn, nk_subst_fn, basis_cache, pol_deg=0.0):
    """1つの(波長,角度)組を実際にS4で解き、(R0, R_total, T0, T_total)を返す。
    simの状態（材質・周波数・励振条件）を書き換える副作用がある。
    pol_deg: 偏光角 ψ。電場を s 方向から p 方向へ ψ だけ回した直線偏光（s 振幅 cosψ、
    p 振幅 sinψ）。0 = s 偏光、90 = p 偏光。入射の強さは ψ によらず同じ（|cos|²+|sin|²=1）。
    """
    wl_um = float(wl_nm) / 1000.0
    update_materials(sim, float(wl_nm), material_names, nk_fn_map, nk_env_fn, nk_subst_fn)
    sim.SetFrequency(1.0 / wl_um)
    theta_deg = theta_rad * 180.0 / np.pi
    phi_deg = phi_rad * 180.0 / np.pi
    psi = np.radians(pol_deg)
    sim.SetExcitationPlanewave(IncidenceAngles=(theta_deg, phi_deg),
                                sAmplitude=float(np.cos(psi)), pAmplitude=float(np.sin(psi)))

    _, idx0 = basis_cache.resolve(sim)

    def fluxes():
        top = sim.GetPowerFluxByOrder(Layer='__env_super__', zOffset=0.0)
        bot = sim.GetPowerFluxByOrder(Layer='__subst__', zOffset=0.0)
        inc = top[idx0][0].real
        return (-top[idx0][1].real / inc, sum(-fb[1].real for fb in top) / inc,
                bot[idx0][0].real / inc, sum(fb[0].real for fb in bot) / inc)

    out = fluxes()
    if any(np.isnan(v) for v in out):
        # PolarizationDecomposition（法線ベクトル場）が作れない形状で層の固有値問題が失敗すると
        # NaN になる（全面を覆う領域は simplify_full_cell_regions で事前に除くが、それ以外の
        # 想定外の形状への保険）。この点だけ偏光分解なし（Laurent則、収束は遅いが有限）で解き直す。
        sim.SetOptions(PolarizationDecomposition=False)
        try:
            out = fluxes()
        finally:
            sim.SetOptions(PolarizationDecomposition=True)
    return out


def calc_array_rcwa_s4(wl_nm_ar, theta_rad, phi_rad, pitch_x_um, pitch_y_um,
                        norder, layers, nk_fn_map, nk_env_fn, nk_subst_fn,
                        stop_event=None, progress_cb=None, gamma_deg=90.0, pol_deg=0.0):
    """基板上の2次元周期ナノ構造アレイ（単一入射角）をS4で解き、波長ごとの
    R0(0次/鏡面反射率)・R_total(全伝搬次数の反射率合計)・T0・T_totalを返す。

    pitch_x_um, pitch_y_um, gamma_deg: 格子ベクトルを規定する（lattice_vectors_um
    参照）。gamma_deg=90.0(デフォルト)で正方/矩形格子（pitch_x=pitch_yなら正方）、
    gamma_deg=60.0かつpitch_x_um=pitch_y_umで六方格子（六方最密配置）になる。
    layers: top-first、{'thickness_nm','bg_material','disks':[...]}のリスト
    （build_array_geometry参照。円盤は1層、球はsphere_disk_layers()で複数層の
    staircase近似にする）。
    nk_env_fn, nk_subst_fn: `wl_nm -> complex`の関数（薄膜材質のnk_fn_mapと
    同じ規約）。環境側は真空なら`lambda wl: 1.0+0j`、大気(data/nk/Air.nk)
    なら該当のnkテーブル読み込み関数を渡す。
    stop_event: `threading.Event`（省略可）。波長ループの各点の直前に
    `is_set()`をチェックし、セットされていればそこまでの結果を返して打ち切る
    （UIの「計算を停止」ボタンから使う想定。1波長点の計算の途中では止まらず、
    その点が終わるまでは走り切る——S4のSimulate自体を中断する手段はない
    ため）。
    progress_cb: `callable(done, total, elapsed_sec)`（省略可）。各波長点の
    計算が終わるたびに呼ばれる。UIの進捗バー・所要時間表示に使う想定。

    戻り値: dict with keys 'R0','R_total','T0','T_total'（各[nwl]のfloat配列、
    stop_eventで打ち切った場合は未計算分がnp.nan）、'order_basis'
    （GetBasisSet()が返す(g1,g2)のタプル、デバッグ用）、'n_computed'
    （実際に計算できた点数。stop_eventで打ち切らなければnwlと一致）。
    """
    _require_s4()
    import time as _time

    layers, pitch_x_um, pitch_y_um = reduce_uniform_period(layers, pitch_x_um, pitch_y_um, gamma_deg, norder)
    layers = simplify_full_cell_regions(layers, pitch_x_um, pitch_y_um, gamma_deg)
    material_names = _collect_material_names(layers)
    nwl = len(wl_nm_ar)
    R0 = np.full(nwl, np.nan); Rtot = np.full(nwl, np.nan)
    T0 = np.full(nwl, np.nan); Ttot = np.full(nwl, np.nan)

    sim = S4.New(Lattice=lattice_vectors_um(pitch_x_um, pitch_y_um, gamma_deg), NumBasis=norder)
    build_array_geometry(sim, layers, material_names)
    basis_cache = _BasisCache()

    n_computed = 0
    t_start = _time.time()

    for idx, wl_nm in enumerate(wl_nm_ar):
        if stop_event is not None and stop_event.is_set():
            break

        r0, rtot, t0, ttot = _solve_one_point(
            sim, wl_nm, theta_rad, phi_rad, material_names, nk_fn_map,
            nk_env_fn, nk_subst_fn, basis_cache, pol_deg)
        R0[idx], Rtot[idx], T0[idx], Ttot[idx] = r0, rtot, t0, ttot
        n_computed = idx + 1

        if progress_cb is not None:
            progress_cb(n_computed, nwl, _time.time() - t_start)

    return dict(R0=R0, R_total=Rtot, T0=T0, T_total=Ttot, order_basis=basis_cache.basis,
                n_computed=n_computed)


def calc_array_rcwa_s4_angle_sweep(wl_nm_ar, theta_rad_ar, phi_rad, pitch_x_um, pitch_y_um,
                                    norder, layers, nk_fn_map, nk_env_fn, nk_subst_fn,
                                    stop_event=None, progress_cb=None, gamma_deg=90.0, pol_deg=0.0):
    """calc_array_rcwa_s4()の角度掃引版。入射角theta_rad_arの配列を受け取り、
    S4のsim/ジオメトリ構築は1回だけ行って、角度×波長の全ての組について
    材質更新・励振条件の再設定だけをやり直す（sim再構築は角度ごとには
    行わない——AddLayer/SetRegionCircle等は角度に依存しないため、
    やり直す必要が無く、コストも高い）。

    gamma_deg: calc_array_rcwa_s4()と同じ（格子ベクトル間の角度。90.0=正方/矩形、
    60.0かつpitch_x=pitch_yで六方格子）。pol_deg: 偏光角 ψ（_solve_one_point 参照）。

    戻り値: dict with keys 'R0','R_total','T0','T_total'が各
    [n_theta, nwl]の2次元float配列（stop_eventで打ち切った場合は
    未計算分がnp.nan）、'order_basis'、'n_computed'（角度×波長の合計に対する
    実際の計算済み点数）。ループの順序は角度が外側・波長が内側
    （koumyouアプリ(R8_koumyou_v4.py)の角度掃引と同じ規約）。
    """
    _require_s4()
    import time as _time

    layers, pitch_x_um, pitch_y_um = reduce_uniform_period(layers, pitch_x_um, pitch_y_um, gamma_deg, norder)
    layers = simplify_full_cell_regions(layers, pitch_x_um, pitch_y_um, gamma_deg)
    material_names = _collect_material_names(layers)
    n_theta = len(theta_rad_ar)
    nwl = len(wl_nm_ar)
    R0 = np.full((n_theta, nwl), np.nan); Rtot = np.full((n_theta, nwl), np.nan)
    T0 = np.full((n_theta, nwl), np.nan); Ttot = np.full((n_theta, nwl), np.nan)

    sim = S4.New(Lattice=lattice_vectors_um(pitch_x_um, pitch_y_um, gamma_deg), NumBasis=norder)
    build_array_geometry(sim, layers, material_names)
    basis_cache = _BasisCache()

    total = n_theta * nwl
    n_computed = 0
    t_start = _time.time()

    for ia, theta_rad in enumerate(theta_rad_ar):
        for iw, wl_nm in enumerate(wl_nm_ar):
            if stop_event is not None and stop_event.is_set():
                return dict(R0=R0, R_total=Rtot, T0=T0, T_total=Ttot,
                            order_basis=basis_cache.basis, n_computed=n_computed)

            r0, rtot, t0, ttot = _solve_one_point(
                sim, wl_nm, theta_rad, phi_rad, material_names, nk_fn_map,
                nk_env_fn, nk_subst_fn, basis_cache, pol_deg)
            R0[ia, iw], Rtot[ia, iw], T0[ia, iw], Ttot[ia, iw] = r0, rtot, t0, ttot
            n_computed += 1

            if progress_cb is not None:
                progress_cb(n_computed, total, _time.time() - t_start)

    return dict(R0=R0, R_total=Rtot, T0=T0, T_total=Ttot, order_basis=basis_cache.basis,
                n_computed=n_computed)


def sphere_disk_layers(radius_nm, material, n_slices, bg_material=ENV_MATERIAL):
    """球（基板表面z=0に1点で接し、中心がz=radius_nmにある配置）を、
    z方向にn_slices枚の薄い円盤（各層内は断面が変化しないstaircase近似）で
    表現したlayersを返す（calc_array_rcwa_s4/build_array_geometryに
    そのまま渡せる、top-first順）。

    各スライスの半径は、そのスライスの高さ範囲の**中点**でのz=一定断面での
    球の半径 sqrt(R^2 - (z_mid-R)^2) を使う（中点則）。n_slicesを増やすほど
    真の球に収束する——収束の実測確認は__main__セルフテスト参照。
    """
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    slice_h = 2.0 * radius_nm / n_slices
    layers = []
    for i in range(n_slices):  # i=0が最上部(top-first)、i=n_slices-1が基板に接する最下部
        z_hi = 2.0 * radius_nm - i * slice_h
        z_lo = z_hi - slice_h
        z_mid = (z_lo + z_hi) / 2.0
        dz = z_mid - radius_nm
        r_mid = float(np.sqrt(max(radius_nm ** 2 - dz ** 2, 0.0)))
        layers.append({
            'thickness_nm': slice_h, 'bg_material': bg_material,
            'disks': [{'material': material, 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': r_mid}],
        })
    return layers


def hemisphere_disk_layers(radius_nm, material, n_slices, bg_material=ENV_MATERIAL):
    """半球（平らな面を下にして基板表面 z=0 に置いたドーム形、高さ = radius_nm）を、
    z方向に n_slices 枚の薄い円盤で表した layers を返す（top-first）。sphere_disk_layers と
    同じ中点則: 各スライスの半径は、そのスライスの高さ範囲の中点 z_mid での断面半径
    sqrt(R^2 - z_mid^2)。最下段は底面（半径R）に近い円盤になる。"""
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    slice_h = radius_nm / n_slices
    layers = []
    for i in range(n_slices):  # i=0 が頂上側、i=n_slices-1 が基板に接する底面側
        z_mid = radius_nm - (i + 0.5) * slice_h
        r_mid = float(np.sqrt(max(radius_nm ** 2 - z_mid ** 2, 0.0)))
        layers.append({
            'thickness_nm': slice_h, 'bg_material': bg_material,
            'disks': [{'material': material, 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': r_mid}],
        })
    return layers


def hcylinder_rect_layers(radius_nm, length_nm, axis, material, n_slices, bg_material=ENV_MATERIAL):
    """横に倒したシリンダー（軸が x または y 方向、下面が z=0 に線で接し、中心軸が z=radius_nm）を、
    z方向に n_slices 枚の薄い矩形で表した layers を返す（top-first）。sphere_disk_layers と同じ
    中点則: 各スライスの幅（軸に垂直な方向）は中点 z_mid での弦の長さ 2·sqrt(R^2-(z_mid-R)^2)、
    長さ（軸方向）は length_nm。端面は平ら。length_nm を軸方向の周期と同じにすると、端が隣の
    周期像とぴったりつながった無限に長い棒（1次元の格子）になる。"""
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    if axis not in ('x', 'y'):
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")
    slice_h = 2.0 * radius_nm / n_slices
    layers = []
    for i in range(n_slices):  # i=0 が最上部
        z_mid = 2.0 * radius_nm - (i + 0.5) * slice_h
        hw = float(np.sqrt(max(radius_nm ** 2 - (z_mid - radius_nm) ** 2, 0.0)))
        hx, hy = (0.5 * length_nm, hw) if axis == 'x' else (hw, 0.5 * length_nm)
        layers.append({
            'thickness_nm': slice_h, 'bg_material': bg_material, 'disks': [],
            'rects': [{'material': material, 'cx_nm': 0.0, 'cy_nm': 0.0, 'hw_x_nm': hx, 'hw_y_nm': hy}],
        })
    return layers


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _require_s4()
    from RCWA2D.material_spec import resolve_nk_fn

    nk_tio2 = resolve_nk_fn("TiO2")
    nk_sio2 = resolve_nk_fn("SiO2")
    nk_env_vacuum = resolve_nk_fn(1.0)   # 真空: 数値指定(定数関数)
    nk_env_air = resolve_nk_fn("Air")    # 大気: data/nk/Air.nk 表引き
    wl_ar = np.array([450.0, 550.0, 650.0])
    norder = 49

    print("=== Check 1: 円盤なし(radius=0、build_array_geometryが円盤自体をスキップする"
          "パス)のとき、単純なFresnel反射率(基板)に一致するか ===")
    layers_tiny = [{'thickness_nm': 100.0, 'bg_material': ENV_MATERIAL,
                     'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 0.0}]}]
    res = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.4, 0.4, norder, layers_tiny,
                              {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2)
    for i, wl in enumerate(wl_ar):
        n_s = complex(nk_sio2(wl))
        r_fresnel = abs((1.0 - n_s) / (1.0 + n_s)) ** 2
        print(f"  wl={wl:.0f}nm  R0(array,disk~0)={res['R0'][i]:.6f}  "
              f"R_fresnel(bare subst)={r_fresnel:.6f}  diff={abs(res['R0'][i]-r_fresnel):.2e}")

    print()
    print("=== Check 2: TiO2円盤アレイ(半径150nm, 400nmピッチ正方格子)でエネルギー保存則 ===")
    layers_disk = [{'thickness_nm': 100.0, 'bg_material': ENV_MATERIAL,
                     'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}]
    res2 = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.4, 0.4, norder, layers_disk,
                               {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2)
    for i, wl in enumerate(wl_ar):
        rt, tt = res2['R_total'][i], res2['T_total'][i]
        print(f"  wl={wl:.0f}nm  R0={res2['R0'][i]:.4f}  R_total={rt:.4f}  "
              f"T0={res2['T0'][i]:.4f}  T_total={tt:.4f}  R+T={rt+tt:.6f}  "
              f"norder(basis)={len(res2['order_basis'])}")

    print()
    print("=== Check 3: 環境=真空(n=1定数関数) vs 環境=大気(Air.nk表引き) が一致するか ===")
    res3 = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.4, 0.4, norder, layers_disk,
                               {'TiO2': nk_tio2}, nk_env_air, nk_sio2)
    for i, wl in enumerate(wl_ar):
        d_r0 = abs(res2['R0'][i] - res3['R0'][i])
        d_t0 = abs(res2['T0'][i] - res3['T0'][i])
        print(f"  wl={wl:.0f}nm  R0(vacuum)={res2['R0'][i]:.6f}  R0(air)={res3['R0'][i]:.6f}  "
              f"diff={d_r0:.2e}  |  T0(vacuum)={res2['T0'][i]:.6f}  T0(air)={res3['T0'][i]:.6f}  "
              f"diff={d_t0:.2e}")

    print()
    print("=== Check 4: 基板をnkファイルでなく数値で直接指定(n=1.5固定)しても"
          "Fresnel反射率と一致するか ===")
    nk_subst_numeric = resolve_nk_fn(1.5)  # ファイル不要、数値をそのまま定数関数化
    res4 = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.4, 0.4, norder, layers_tiny,
                               {'TiO2': nk_tio2}, nk_env_vacuum, nk_subst_numeric)
    r_fresnel_15 = abs((1.0 - 1.5) / (1.0 + 1.5)) ** 2
    for i, wl in enumerate(wl_ar):
        print(f"  wl={wl:.0f}nm  R0(array, subst=1.5数値指定)={res4['R0'][i]:.6f}  "
              f"R_fresnel(n=1.5)={r_fresnel_15:.6f}  diff={abs(res4['R0'][i]-r_fresnel_15):.2e}")

    print()
    print("=== Check 5: 角度掃引版が単一角度版(calc_array_rcwa_s4)と一致するか"
          "（_solve_one_point共通化のリファクタリング確認） ===")
    theta_ar = np.array([0.0, 20.0, 40.0]) * np.pi / 180.0
    res_sweep = calc_array_rcwa_s4_angle_sweep(wl_ar, theta_ar, 0.0, 0.4, 0.4, norder, layers_disk,
                                                 {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2)
    max_diff = 0.0
    for ia, th in enumerate(theta_ar):
        res_single = calc_array_rcwa_s4(wl_ar, th, 0.0, 0.4, 0.4, norder, layers_disk,
                                          {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2)
        d = np.max(np.abs(res_sweep['R0'][ia, :] - res_single['R0']))
        max_diff = max(max_diff, d)
        print(f"  theta={np.degrees(th):.0f}deg  max|R0(sweep)-R0(single)|={d:.2e}")
    print(f"  全体max diff = {max_diff:.2e}（0に近いほど良い、リファクタリングによる劣化なし）")

    print()
    print("=== Check 6: 球体アレイ(staircase近似)でエネルギー保存則 + n_slices収束 ===")
    for n_slices in (4, 10, 30, 80):
        layers_sphere = sphere_disk_layers(150.0, 'TiO2', n_slices)
        res_sph = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.5, 0.5, norder, layers_sphere,
                                       {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2)
        rt, tt = res_sph['R_total'], res_sph['T_total']
        print(f"  n_slices={n_slices:3d}  R+T={list(np.round(rt+tt, 5))}  "
              f"R0={list(np.round(res_sph['R0'], 4))}")
    print("  ↑ R+Tが1に近いか、n_slicesを増やすとR0の値が収束していくかを確認する"
          "（球全体がピッチ0.5umの中に収まるよう半径150nmに対して余裕を持たせた格子）。")

    print()
    print("=== Check 7: 六方格子(gamma_deg=60, pitch_x=pitch_y)でエネルギー保存則 ===")
    layers_hex = [{'thickness_nm': 100.0, 'bg_material': ENV_MATERIAL,
                    'disks': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0, 'radius_nm': 150.0}]}]
    res_hex = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.4, 0.4, norder, layers_hex,
                                   {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2, gamma_deg=60.0)
    for i, wl in enumerate(wl_ar):
        rt, tt = res_hex['R_total'][i], res_hex['T_total'][i]
        print(f"  wl={wl:.0f}nm  R0={res_hex['R0'][i]:.4f}  R_total={rt:.4f}  "
              f"T0={res_hex['T0'][i]:.4f}  T_total={tt:.4f}  R+T={rt+tt:.6f}")

    print()
    print("=== Check 8: 六方格子のC6対称性(φ=0°とφ=60°で入射条件を60°回転させても"
          "R0/R_totalが一致するはず) ===")
    theta_oblique = 30.0 * np.pi / 180.0
    res_phi0 = calc_array_rcwa_s4(wl_ar, theta_oblique, 0.0, 0.4, 0.4, norder, layers_hex,
                                    {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2, gamma_deg=60.0)
    res_phi60 = calc_array_rcwa_s4(wl_ar, theta_oblique, 60.0 * np.pi / 180.0, 0.4, 0.4, norder,
                                     layers_hex, {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2, gamma_deg=60.0)
    max_diff_hex = np.max(np.abs(res_phi0['R0'] - res_phi60['R0']))
    for i, wl in enumerate(wl_ar):
        print(f"  wl={wl:.0f}nm  R0(phi=0)={res_phi0['R0'][i]:.6f}  "
              f"R0(phi=60)={res_phi60['R0'][i]:.6f}  diff={abs(res_phi0['R0'][i]-res_phi60['R0'][i]):.2e}")
    print(f"  全体max diff = {max_diff_hex:.2e}（0に近いほど良い、六方格子のC6対称性が正しく"
          "実装されている証拠。正方格子(gamma_deg=90)ならこの対称性はC4なのでφ=0/60では"
          "一致しないはず——比較用に併記）")
    res_sq_phi0 = calc_array_rcwa_s4(wl_ar, theta_oblique, 0.0, 0.4, 0.4, norder, layers_hex,
                                       {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2, gamma_deg=90.0)
    res_sq_phi60 = calc_array_rcwa_s4(wl_ar, theta_oblique, 60.0 * np.pi / 180.0, 0.4, 0.4, norder,
                                        layers_hex, {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2, gamma_deg=90.0)
    print(f"  (参考)正方格子 max|R0(phi=0)-R0(phi=60)| = "
          f"{np.max(np.abs(res_sq_phi0['R0']-res_sq_phi60['R0'])):.2e}（0でないのが正常）")

    print()
    print("=== Check 9: 矩形領域(SetRegionRectangle、段付き多層膜の基板上げ領域用)が"
          "2D格子でも正しく動くか ===")
    print("  --- 9a: 半幅がほぼ0の極限でFresnel反射率(基板)に一致するか ---")
    layers_rect_tiny = [{'thickness_nm': 100.0, 'bg_material': ENV_MATERIAL, 'disks': [],
                          'rects': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0,
                                     'hw_x_nm': 1e-6, 'hw_y_nm': 1e-6}]}]
    res_rect_tiny = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.4, 0.4, norder, layers_rect_tiny,
                                         {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2)
    for i, wl in enumerate(wl_ar):
        n_s = complex(nk_sio2(wl))
        r_fresnel = abs((1.0 - n_s) / (1.0 + n_s)) ** 2
        print(f"  wl={wl:.0f}nm  R0(rect~0)={res_rect_tiny['R0'][i]:.6f}  "
              f"R_fresnel(bare subst)={r_fresnel:.6f}  diff={abs(res_rect_tiny['R0'][i]-r_fresnel):.2e}")

    print("  --- 9b: 通常サイズの矩形領域(半幅150nm=一辺300nm、ピッチ400nm)でエネルギー保存則 ---")
    layers_rect = [{'thickness_nm': 100.0, 'bg_material': ENV_MATERIAL, 'disks': [],
                     'rects': [{'material': 'TiO2', 'cx_nm': 0.0, 'cy_nm': 0.0,
                                'hw_x_nm': 150.0, 'hw_y_nm': 150.0}]}]
    res_rect = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.4, 0.4, norder, layers_rect,
                                    {'TiO2': nk_tio2}, nk_env_vacuum, nk_sio2)
    for i, wl in enumerate(wl_ar):
        rt, tt = res_rect['R_total'][i], res_rect['T_total'][i]
        print(f"  wl={wl:.0f}nm  R0={res_rect['R0'][i]:.4f}  R_total={rt:.4f}  "
              f"T0={res_rect['T0'][i]:.4f}  T_total={tt:.4f}  R+T={rt+tt:.6f}")

    print("  --- 9c: structure_builder.regions_to_layers_2d()で組んだ段付き多層膜"
          "（中央に基板上げ矩形領域、周囲はTiO2/SiO2ペア2回）でもエネルギー保存則 ---")
    from RCWA2D.structure_builder import regions_to_layers_2d  # sys.path挿入済み(本ブロック冒頭)
    _field_recipe = [{'type': 'pair', 'mat_a': 'TiO2', 'th_a': 51.0,
                       'mat_b': 'SiO2', 'th_b': 81.0, 'n_pairs': 2}]
    _regions = [{'shape': 'rect', 'cx_nm': 0.0, 'cy_nm': 0.0,
                 'hw_x_nm': 100.0, 'hw_y_nm': 100.0, 'raise_nm': 50.0}]
    layers_stepped = regions_to_layers_2d(_regions, _field_recipe)
    res_stepped = calc_array_rcwa_s4(wl_ar, 0.0, 0.0, 0.5, 0.5, norder, layers_stepped,
                                       {'TiO2': nk_tio2, 'SiO2': nk_sio2}, nk_env_vacuum, nk_sio2)
    for i, wl in enumerate(wl_ar):
        rt, tt = res_stepped['R_total'][i], res_stepped['T_total'][i]
        print(f"  wl={wl:.0f}nm  R0={res_stepped['R0'][i]:.4f}  R_total={rt:.4f}  "
              f"T0={res_stepped['T0'][i]:.4f}  T_total={tt:.4f}  R+T={rt+tt:.6f}  "
              f"layers={len(layers_stepped)}")
    print("  ↑ 9a/9bのR+Tが1に近く、9aがFresnel値と一致すれば、SetRegionRectangleが"
          "2D格子でも1D backendと同様に正しく動作していると言える。9cはそれを実際の"
          "段付き多層膜ジオメトリ(regions_to_layers_2d)に適用した統合テスト。")

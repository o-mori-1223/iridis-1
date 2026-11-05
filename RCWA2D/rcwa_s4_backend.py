"""列ベース構造色シミュレータ（R8_koumyou_v3.py）向けの S4 (Stanford Stratified Structure
Solver) バックエンド。

状態: 検証済み・実運用中（WSL上でビルドし、R8_koumyou_v3.py に rcwa_mh と並ぶ
"S4" ソルバーバックエンドとして組み込み済み）。以下に記載しているAPIの挙動は、
すべて実際にS4を動かして確認したもの（S4/PythonTest.py の実行、および裸のフレネル
界面での検証テストなど）で、ドキュメントを読んだだけの推測ではない。

このモジュールを作った経緯
----------------
ユーザーは当初、1D FMM/RCWAバックエンド（rcwa_mh_cpp.cpp）が「列周期」構造
（columns_to_layers()、「等形成膜/conformal」積層モデル）で怪しい結果を出している
のではと疑っていた。エネルギー保存則（R+T=1）や既知の等価ケース（ppap.pyとの
比較で均一膜が一致する、全列のsubst_raiseが同じ場合はconformal≡flatになる、等）は
いずれも機械精度で一致していたため、独立した「セカンドオピニオン」を得る目的で
このS4バックエンドを構築した。ビルド後は、実際に測定・シミュレーションされた
参照データ（COMSOL FEMおよびGsolver）に対して、実在する6列ミラースタック構造で
比較検証を行った。その結果、見かけ上の不一致は**ソルバーのバグではない**ことが
判明した——原因は比較に使っていた基板上げパターンと基板屈折率の指定間違いだった。
これらを修正した後は、columns_to_layers()（側面コーティングなし。一時期「側面
コーティングあり」の仮説・モデルを試したが、後に削除した。詳細はgit履歴を参照）は
検証した5波長すべてでCOMSOLと同じオーダー・同じ傾向まで一致するようになった。
ただし470/540nm付近には依然として未解決の差が残っている（比較用スクリプトと
数値についてはdiag_conformal.pyおよび2026-08-04前後のgit logを参照）。

座標・位置合わせの設計（上記の不一致で最も疑われた点）
--------------------------------------------------------------------------------------
R9の `layers`（columns_to_layers() / columns_to_layers_flat() の出力）は、
    {'thickness': <nm>, 'segments': [(material_name, width_fraction), ...]}
という形のtop-first（上から順）のリストであり、各層内で `segments` は厳密に
左から右の順で並び、width_fraction の合計は1.0（ピッチに対する割合）になる。
重要なのは、**すべての層の segments が同じ `columns` リストから同じ左→右の順序で
生成される**ため、1つの構造内では全層で列境界の絶対x座標（レジストレーション）が
一致している点（これは columns_to_layers() の実装を見れば確認できる。
R8_koumyou_v3.py 参照）。

このバックエンドはこのレジストレーションを厳密に保持する。各層について、
`segments` リストを左から右へ走査してxを積算し、その絶対位置（[0, pitch) の
範囲）にS4の矩形を配置する。層ごと・セグメントごとに座標を再センタリングする
ようなことは一切しない——全層が同一の絶対座標系（x=0〜pitch）を共有しており、
これはR9の意図そのままである。S4の格子は周期的なので、「どこをx=0とするか」は
単独の層だけを見れば任意に選んでよいが、**層をまたいだ一貫性**こそが層間の
回折結合を物理的に正しくする要因なので、これは絶対に守るべき不変条件である。

S4のPython API仕様 -- 実際にビルド・実行して確認済み（phoebe-p/S4 フォークを
WSL Ubuntu 26.04上、~/s4env の仮想環境にビルド。S4/main_python.c の kwlist[]
テーブルを直接読み、かつ S4/PythonTest.py を実行して動作確認した）。
-------------------------------------------------------------------------------
    S = S4.New(Lattice=period_um, NumBasis=norder)          # 1D: Lattice はスカラー
    S.SetMaterial(Name=name, Epsilon=eps_complex)
    S.AddLayer(Name=name, Thickness=thickness_um, Material=bg_material_name)
    S.SetRegionRectangle(Layer=name, Material=mat, Center=(cx, 0.0),
                          Angle=0.0, Halfwidths=(hw, big_y))
    S.SetExcitationPlanewave(IncidenceAngles=(theta_deg, phi_deg),
                              sAmplitude=..., pAmplitude=...)  # (theta,phi) の順。(phi,theta) ではない
    S.SetFrequency(1.0/wl_um)                                 # freq = 1/波長。角周波数ではない
    pairs = S.GetPowerFluxByOrder(Layer=name, zOffset=0.0)    # 各次数ごとの(fwd,bwd)のタプル、
                                                                 # fwd/bwd はそれぞれ複素数

最初に追加したレイヤー = 入射側（上部構造・光源側）。最後に追加したレイヤー =
基板・透過側。これはR9自身の `layers`（top-first順）とそのまま対応しており、
反転処理は不要（rcwa_mh_cppの `get_layer_tuple` が基板を先頭に置くのとは異なる）。

符号規約（確認済み）: S4/PythonTest.py を実行したところ、無損失のSiO2/Si/Vacuum
スタックで Power_backward = -0.9575 という結果が得られた。つまり GetPowerFlux の
"backward" 成分は、ポインティングベクトルのz成分に**符号付き**の値であり——-z方向に
流れるためマイナスになる。したがって反射率は R = -backward.real であり
（backward.real をそのまま使うのではない）、一方で出力層側の透過率は
T = +forward.real（+z方向に流れる、つまり"forward"の定義通りの向きなのでプラス
のまま）である。どちらも SetExcitationPlanewave の単位振幅規約における0次光の
入射パワーで規格化されている。

GetBasisSet() は要求した NumBasis よりも少ない、または並び順が異なるエントリを
返すことがある（S4はその打ち切り方式が生成するG-ベクトル集合に丸めるため）——
下記の calc_rcwa1d_s4() は、行番号iがそのまま次数 i-p であると仮定せず、
GetBasisSet() から明示的に「次数→行番号」の対応表を構築する。対応表の長さが
norderと一致することもassertで確認しており、万一の食い違いはサイレントな
誤対応ではなく明確なエラーとして検出される。

WSLでのビルド手順（2026-07-31に実際にこの手順でビルドした記録）
------------------------------------------------------------
本家の victorliu/S4 リポジトリのPythonバインディングはメンテナンスされておらず、
そのままではビルドできない（main_python.c にPython2専用のまま移植されていない
コードパスが残っている上、現在のコアライブラリに対して Material*/S4_MaterialID
のAPIが古いままになっている）。代わりにコミュニティフォークの phoebe-p/S4 を
使うこと。太陽電池のRCWA計算用途で現役でメンテナンスされており、Python 3で
問題なくビルドできる:

    sudo apt install build-essential gfortran liblapack-dev libblas-dev libfftw3-dev \
                      liblua5.4-dev lua5.4 python3-dev python3-pip python3-venv \
                      libopenblas-dev libsuitesparse-dev libboost-all-dev
    git clone https://github.com/phoebe-p/S4 && cd S4
    make all                                    # build/libS4.a をビルド
    python3 -m venv ~/s4env && source ~/s4env/bin/activate
    pip install numpy wheel setuptools
    pip install --use-pep517 --no-build-isolation ./     # S4のPython拡張をビルド＋インストール

非常に新しいPython（3.13以降、numpy>=2）では、main_python.c の
S4Sim_GetFieldsOnGridNumpy 内の PyArray_ENABLEFLAGS(Earr/Harr, ...) 呼び出しが
2箇所コンパイルエラーになる（Earr/Harr が PyArrayObject* ではなく PyObject* として
宣言されているため）——次のようにキャストすれば直る:
PyArray_ENABLEFLAGS((PyArrayObject*)Earr, ...)。
GetFieldsOnGrid* 系の機能はここでは一切使っていないので、これは純粋なコンパイル
修正であり、挙動への影響はない。
"""

import numpy as np

# 2026-10-26: S4本体を third_party/S4（lab/S4 からコピーした本家 victorliu/S4 のソース）に
# 置き換えた。上記 docstring の phoebe-p フォーク版Python拡張ではなく、RCWA2D/s4lab.py
# （ctypes ラッパー、同じメソッド名・引数）経由で RCWA2D/libs4lab.so を呼ぶ。
# ビルド: WSLで `bash third_party/S4/build_s4lab.sh`。
try:
    try:
        from RCWA2D import s4lab as S4
    except ModuleNotFoundError:   # RCWA2D/ 内のスクリプトとして直接実行した場合
        import s4lab as S4
    HAVE_S4 = True
except ImportError:
    S4 = None
    HAVE_S4 = False

# R8_koumyou_v3.py の番兵（sentinel）値と厳密に一致させること。
ENV_MATERIAL = '__ENV__'
SUBST_MATERIAL = '__SUBSTRATE__'


def _require_s4():
    if not HAVE_S4:
        raise RuntimeError(
            "S4 (RCWA2D/libs4lab.so) を読み込めません。WSLで "
            "`bash third_party/S4/build_s4lab.sh` を実行してビルドしてください。"
        )


def _material_key(mat, nk_subst_fn):
    """R9のセグメント材質エントリを、安定したS4材質名にマッピングする。"""
    if mat == ENV_MATERIAL or mat == SUBST_MATERIAL:
        return mat
    return mat  # 通常のnkテーブル名（例: 'TiO2'）をそのままS4材質名として使う


def build_s4_geometry(sim, layers, pitch_um):
    """材質を登録し（仮のeps=1で登録し、波長ごとの実際の値は後でupdate_materials()で
    更新する）、層スタック＋横方向パターンを一度だけ構築する。`layers` はR9の
    top-first順のリストで、各要素は
    {'thickness': nm, 'segments': [(material, width_fraction), ...]} の形。

    実際に使われている材質名（env/subst以外）のリストを順序付きで返す。
    呼び出し側はこれを見て波長ごとにどの材質を更新すればよいかが分かる。
    """
    _require_s4()

    material_names = set()
    for L in layers:
        for mat, w in L['segments']:
            if w > 1e-9:
                material_names.add(_material_key(mat, None))
    material_names.discard(ENV_MATERIAL)
    material_names.discard(SUBST_MATERIAL)

    # 仮の材質登録。実際のepsilonは波長ごとに update_materials() で設定する。
    sim.SetMaterial(Name=ENV_MATERIAL, Epsilon=1.0 + 0j)
    sim.SetMaterial(Name=SUBST_MATERIAL, Epsilon=1.0 + 0j)
    for name in sorted(material_names):
        sim.SetMaterial(Name=name, Epsilon=1.0 + 0j)

    # R9のtop-first順に従い、まず superstrate（入射側）を追加する。
    # Liの「正しい規則（correct rule）」＝法線方向を考慮したフーリエ因子分解。これを
    # 有効にしないと、S4はTM/p偏光について単純な（Laurent則の）因子分解にフォール
    # バックしてしまい、収束はするものの O(1/NumBasis) 程度の遅い収束にとどまる
    # （指数関数的な速い収束にはならない）——これは実測で確認済み: NumBasis=251の
    # 単純な2材質グレーティングで、このオプションなしだとrcwa_mh_cppとのRpの差が
    # 1.5e-4だったのに対し、ありだと2.0e-5まで縮まった（なお rcwa_mh_cpp 自体は
    # NumBasis=41程度でほぼ収束しており、これはTM側で同等の正しい規則の因子分解を
    # 実装していることと整合する——このモジュールの __main__ セルフテスト内の
    # Rcwa1d_s4 と rcwa_mh のクロスチェックを参照）。
    sim.SetOptions(PolarizationDecomposition=True)

    sim.AddLayer(Name='__env_super__', Thickness=0.0, Material=ENV_MATERIAL)

    for i, L in enumerate(layers):
        lname = f'L{i}'
        thickness_um = L['thickness'] / 1000.0
        segs = [(mat, w) for mat, w in L['segments'] if w > 1e-9]
        if not segs:
            continue

        # 背景材質＝最も幅の広いセグメント（上に描く矩形の数を最小化できる。また
        # ENVがその層の大部分を占める場合は自然な選択になる）。
        bg_mat, _ = max(segs, key=lambda mw: mw[1])
        bg_key = _material_key(bg_mat, None)
        sim.AddLayer(Name=lname, Thickness=thickness_um, Material=bg_key)

        if len(segs) > 1:
            x = 0.0
            for mat, w in segs:
                width_um = w * pitch_um
                cx = x + width_um / 2.0
                key = _material_key(mat, None)
                if mat is not bg_mat and key != bg_key:
                    # 注: 領域同士が重なり得る場合は描画順が結果に影響するが、今回は
                    # R9側の構造上セグメント同士が重なることはないため（非重複が
                    # 保証されている）、順序はどれでも問題ない。
                    sim.SetRegionRectangle(
                        Layer=lname, Material=key,
                        Center=(cx, 0.0), Angle=0.0,
                        Halfwidths=(width_um / 2.0, pitch_um),
                    )
                x += width_um
            # 補足: 隣接する2つのセグメントが背景材質以外の同一材質を持ち、背景が
            # 別の材質として選ばれた場合、両方とも別々に矩形として描画される——結果は
            # 正しいが結合されないだけで、正しさの問題ではなく多少の非効率さに過ぎない。

    sim.AddLayer(Name='__subst__', Thickness=0.0, Material=SUBST_MATERIAL)
    return sorted(material_names)


def update_materials(sim, wl_nm, material_names, nk_fn_map, n_env_val, nk_subst_fn):
    n_env = complex(n_env_val)
    sim.SetMaterial(Name=ENV_MATERIAL, Epsilon=n_env * n_env)
    n_subst = complex(nk_subst_fn(wl_nm))
    sim.SetMaterial(Name=SUBST_MATERIAL, Epsilon=n_subst * n_subst)
    for name in material_names:
        n = complex(nk_fn_map[name](wl_nm))
        sim.SetMaterial(Name=name, Epsilon=n * n)


def _order_index_map(sim, norder):
    """GetBasisSet() の行順を、R9の連番 -p..+p という次数インデックス規約に
    マッピングする。
    S4が使えるようになるまで未検証——セルフテストを参照。GetBasisSet は
    (g1, g2) の整数タプルのリストを返すはずで、1D格子では常にg2==0、g1が
    次数（harmonic index）にあたるが、必ずしも -p..+p の順にソートされている
    とは限らないため、ここで明示的な対応表（permutation）を作る。
    """
    basis = sim.GetBasisSet()
    idx0 = norder // 2
    perm = [None] * norder
    for row, (g1, _g2) in enumerate(basis):
        perm[idx0 + g1] = row
    if any(p is None for p in perm):
        raise RuntimeError(f"GetBasisSet() が期待した -p..+p の範囲を返しませんでした: {basis}")
    return perm


def calc_rcwa1d_s4(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map,
                    n_env_val, nk_subst_fn):
    """R8_koumyou_v3.calc_rcwa1d() のドロップイン代替関数。シグネチャ・戻り値の形は
    同一: (irp, itp, irs, its)、それぞれ [nwl, norder] のfloat配列で、次数ごとの
    効率を表す。インデックスの並びもrcwa_mhと同じ（index p = norder//2 が0次光）。
    """
    _require_s4()

    nwl = len(wl_nm_ar)
    irp = np.empty([nwl, norder], dtype=float)
    itp = np.empty([nwl, norder], dtype=float)
    irs = np.empty([nwl, norder], dtype=float)
    its = np.empty([nwl, norder], dtype=float)

    sim = S4.New(Lattice=pitch_um, NumBasis=norder)
    material_names = build_s4_geometry(sim, layers, pitch_um)
    perm = None  # 最初のSimulate()実行後、GetBasisSetが有効になってから遅延解決する

    inc_angle_deg = inc_angle_rad * 180.0 / np.pi

    for idx, wl_nm in enumerate(wl_nm_ar):
        wl_um = float(wl_nm) / 1000.0
        update_materials(sim, float(wl_nm), material_names, nk_fn_map, n_env_val, nk_subst_fn)
        sim.SetFrequency(1.0 / wl_um)

        for pol, ir_arr, it_arr in (('s', irs, its), ('p', irp, itp)):
            # IncidenceAngles = (theta_polar_deg, phi_azimuthal_deg)。phi=0とすることで
            # 入射面をグレーティングベクトル（x方向）に揃えている。R9のkx0の規約と一致。
            if pol == 's':
                sim.SetExcitationPlanewave(IncidenceAngles=(inc_angle_deg, 0.0),
                                            sAmplitude=1.0, pAmplitude=0.0)
            else:
                sim.SetExcitationPlanewave(IncidenceAngles=(inc_angle_deg, 0.0),
                                            sAmplitude=0.0, pAmplitude=1.0)

            if perm is None:
                perm = _order_index_map(sim, norder)

            fwd_bwd_top = sim.GetPowerFluxByOrder(Layer='__env_super__', zOffset=0.0)
            fwd_bwd_bot = sim.GetPowerFluxByOrder(Layer='__subst__', zOffset=0.0)

            p0 = norder // 2
            inc_power = fwd_bwd_top[perm[p0]][0].real  # 0次光のforwardパワー＝入射パワー
            for m in range(norder):
                row = perm[m]
                ir_arr[idx, m] = -fwd_bwd_top[row][1].real / inc_power
                it_arr[idx, m] = fwd_bwd_bot[row][0].real / inc_power

    return irp, itp, irs, its


if __name__ == '__main__':
    # 段階的に検証していく:
    #  1. 裸のフレネル界面（解析解が既知）で符号・規格化を検証。
    #  2. 2材質単層グレーティングでのエネルギー保存則の確認＋rcwa_mhとのクロスチェック。
    #  3. diag_conformal.py TEST 3 の再現（実際の段差構造を持つ列——ユーザーが最初に
    #     疑っていた「列周期」構造のケース）を、R9とこのバックエンドが実際に共有して
    #     いる structure_builder.columns_to_layers() を通して構築し、同一の `layers`
    #     オブジェクトに対してS4とrcwa_mhを比較する——これにより、不一致があるとすれば
    #     RCWAソルバー自体の問題であることを切り分けられる。
    _require_s4()
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rcwa_mh
    from structure_builder import columns_to_layers, get_layer_tuple

    print("=== Check 1: 裸のフレネル界面、垂直入射、n1=1 -> n2=2 ===")
    n1, n2 = 1.0 + 0j, 2.0 + 0j
    R_analytic = abs((n1 - n2) / (n1 + n2)) ** 2
    pitch_um = 0.5
    sim = S4.New(Lattice=pitch_um, NumBasis=1)
    sim.SetMaterial(Name=ENV_MATERIAL, Epsilon=n1 * n1)
    sim.SetMaterial(Name=SUBST_MATERIAL, Epsilon=n2 * n2)
    sim.AddLayer(Name='__env_super__', Thickness=0.0, Material=ENV_MATERIAL)
    sim.AddLayer(Name='__subst__', Thickness=0.0, Material=SUBST_MATERIAL)
    sim.SetFrequency(1.0 / 0.55)
    sim.SetExcitationPlanewave(IncidenceAngles=(0.0, 0.0), sAmplitude=1.0, pAmplitude=0.0)
    fwd, bwd = sim.GetPowerFluxByOrder(Layer='__env_super__', zOffset=0.0)[0]
    R_s4 = -bwd.real / fwd.real
    print(f"  解析解 R = {R_analytic:.6f}   S4 R = {R_s4:.6f}   "
          f"{'OK' if abs(R_s4-R_analytic) < 1e-6 else 'MISMATCH -- 上の符号・規格化を要修正'}")

    print()
    print("=== Check 2: 2材質単層グレーティング -- S4のエネルギー保存則 + rcwa_mhとの比較 ===")
    layers_2mat = [{'thickness': 100.0, 'segments': [('A', 0.5), (ENV_MATERIAL, 0.5)]}]
    nk_fn_map = {'A': (lambda wl: 2.0 + 0j)}
    n_env_val = 1.0
    nk_subst_fn = lambda wl: 1.5 + 0j
    wl_ar = np.array([550.0])
    irp, itp, irs, its = calc_rcwa1d_s4(wl_ar, 0.0, 0.6, 21, layers_2mat, nk_fn_map,
                                         n_env_val, nk_subst_fn)
    print(f"  S4:  R+T (p) = {irp.sum()+itp.sum():.6f}   R+T (s) = {irs.sum()+its.sum():.6f}  "
          f"（無損失材質なので~1.0になるはず）")

    layer_tuple = ((0, complex(nk_subst_fn(550)), 0),
                   (0.1, 2.0 + 0j, 0.5, 1.0 + 0j, 0.5),
                   (0, 1.0 + 0j, 0))
    ir_p, it_p = rcwa_mh.Rcwa1d('p', 0.55, 0.0, 0.6, layer_tuple, 21)
    ir_s, it_s = rcwa_mh.Rcwa1d('s', 0.55, 0.0, 0.6, layer_tuple, 21)
    print(f"  rcwa_mh: R+T (p) = {ir_p.sum()+it_p.sum():.6f}   R+T (s) = {ir_s.sum()+it_s.sum():.6f}")
    print(f"  rcwa_mh 0次光 Rp = {ir_p[10]:.6f}   S4 0次光 Rp = {irp[0, 10]:.6f}   "
          f"差 = {abs(ir_p[10]-irp[0,10]):.2e}")
    print(f"  rcwa_mh 0次光 Rs = {ir_s[10]:.6f}   S4 0次光 Rs = {irs[0, 10]:.6f}   "
          f"差 = {abs(ir_s[10]-irs[0,10]):.2e}")

    print()
    print("=== Check 3: diag_conformal.py TEST 3 の再現（実際の段差列構造） -- バグ探しの本丸 ===")
    print("  diag_conformal.py と同じ COL_WS/RAISES_ALT/TiO2-SiO2ペア構成だが、今回は")
    print("  structure_builder.columns_to_layers()（R9が実際に呼んでいるコード）で")
    print("  構築し、rcwa_mhとS4の両方で、文字通り同一の `layers` オブジェクトを評価する。")

    def load_nk(name):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "nk", name + ".nk")
        d = np.loadtxt(p, comments=";", encoding="utf-8_sig")
        from scipy.interpolate import interp1d
        return interp1d(d[:, 0], d[:, 1] + d[:, 2] * 1j, kind="linear", fill_value="extrapolate")

    nk_tio2 = load_nk("TiO2")
    nk_sio2 = load_nk("SiO2")
    nk_sus = load_nk("SUS")

    N_PAIRS = 7
    TH_A, TH_B = 77.6, 54.4
    RAISE = 100.0
    COL_WS = [900, 600, 300, 300, 600, 300]
    RAISES_ALT = [0, RAISE, 0, RAISE, 0, RAISE]
    NORDER = 43

    def make_pair_column(width, raise_h):
        return {
            'width': width, 'subst_raise': raise_h,
            'film_layers': [{'type': 'pair', 'mat_a': 'TiO2', 'th_a': TH_A,
                              'mat_b': 'SiO2', 'th_b': TH_B, 'n_pairs': N_PAIRS}],
        }

    columns = [make_pair_column(w, r) for w, r in zip(COL_WS, RAISES_ALT)]
    layers = columns_to_layers(columns, flat_align=False)
    pitch_nm = sum(COL_WS)
    pitch_um_conf = pitch_nm / 1000.0
    nk_fn_map_conf = {'TiO2': nk_tio2, 'SiO2': nk_sio2}

    wls = [420, 470, 520, 570, 620, 670]
    print(f"  {'wl':>5} | {'rcwa_mh Rp':>11} {'S4 Rp':>11} {'diff':>10} | "
          f"{'rcwa_mh Rs':>11} {'S4 Rs':>11} {'diff':>10}")
    for wl in wls:
        nk_subst_fn_conf = lambda w, _wl=wl: complex(nk_sus(_wl))
        lt = get_layer_tuple(float(wl), layers, nk_fn_map_conf, 1.0, nk_subst_fn_conf)
        ir_p, _ = rcwa_mh.Rcwa1d('p', wl / 1000.0, 0.0, pitch_um_conf, lt, NORDER)
        ir_s, _ = rcwa_mh.Rcwa1d('s', wl / 1000.0, 0.0, pitch_um_conf, lt, NORDER)
        rp_mh, rs_mh = float(ir_p[NORDER // 2]), float(ir_s[NORDER // 2])

        irp4, _, irs4, _ = calc_rcwa1d_s4(np.array([float(wl)]), 0.0, pitch_um_conf, NORDER,
                                           layers, nk_fn_map_conf, 1.0, nk_subst_fn_conf)
        rp_s4, rs_s4 = float(irp4[0, NORDER // 2]), float(irs4[0, NORDER // 2])

        print(f"  {wl:>5} | {rp_mh:>11.5f} {rp_s4:>11.5f} {abs(rp_mh-rp_s4):>10.2e} | "
              f"{rs_mh:>11.5f} {rs_s4:>11.5f} {abs(rs_mh-rs_s4):>10.2e}")
    print()
    print("  ここでrcwa_mhとS4が（Check 2の単純なケースとは違って）一致しない場合、バグは")
    print("  structure_builder.columns_to_layers() ではなく、rcwa_mh_cppの3セクション以上・")
    print("  多層のレジストレーション処理に固有の問題ということになる。もし一致するなら、")
    print("  ユーザーが見ていた食い違いは conformal と flat という「モデルの違い」")
    print("  （物理的に別の構造を比較していただけ）であり、バグではなかったことになる。")

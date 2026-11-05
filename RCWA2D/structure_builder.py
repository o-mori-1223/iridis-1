"""Column-based structure -> RCWA horizontal-layer conversion.

Extracted verbatim from R8_koumyou_v3.py so that both the existing rcwa_mh_cpp backend and
the new S4 backend (rcwa_s4_backend.py) consume the EXACT SAME structure-building code. This
matters specifically because the user has flagged the "column-periodic" (列周期, i.e.
multiple side-by-side columns each with an independent film stack -- the "等形成膜/conformal"
deposition model) results as suspect: if the two backends ever disagreed, having them share
this module means the disagreement can only come from the RCWA solvers themselves, not from
two independently-transcribed (and possibly divergently-buggy) versions of the geometry
conversion.

【2026-10-23追記】`regions_to_layers_2d()`は上記と同じ考え方を2次元周期アレイ
(rcwa_s4_array2d_backend.py)向けに転用したもの——「列」を「円/矩形の2D領域」に
置き換えただけで、バンド展開のコア(`_expand_to_bands`)は1D/2D共通のまま無改造で
再利用している（基板上げ高さと成膜レシピだけを見る関数なので、列幅のような
1D固有の概念に依存していない）。「段付き(コンフォーマル)多層膜」——基板の一部
領域だけを上げ、その上に同じ成膜レシピを段差に追従させて積む構造——をiridis.pyの
2Dアレイモードで組み立てるために追加した。
"""

import numpy as np

ENV_MATERIAL = '__ENV__'
SUBST_MATERIAL = '__SUBSTRATE__'


def _add_col_segs(segs, mat, fill_frac, align, col_frac):
    """列幅 col_frac の中で material を fill_frac 分だけ配置してsegsに追加する"""
    if mat == ENV_MATERIAL or fill_frac < 1e-9:
        segs.append((ENV_MATERIAL, col_frac))
        return
    if fill_frac >= 1.0 - 1e-9:
        segs.append((mat, col_frac))
        return
    mat_w = fill_frac * col_frac
    env_w = (1.0 - fill_frac) * col_frac
    if align == 'Right':
        if env_w > 1e-9: segs.append((ENV_MATERIAL, env_w))
        segs.append((mat, mat_w))
    elif align == 'Center':
        hw = env_w / 2.0
        if hw > 1e-9: segs.append((ENV_MATERIAL, hw))
        segs.append((mat, mat_w))
        if hw > 1e-9: segs.append((ENV_MATERIAL, hw))
    else:  # Left
        segs.append((mat, mat_w))
        if env_w > 1e-9: segs.append((ENV_MATERIAL, env_w))


def _expand_to_bands(col, flat_align_height=0.0):
    """Column の film_layers を高さバンドリストに展開する。
    flat_align_height > 0 のとき、基板上げ高さ〜flat_align_height を ENV で埋めてから薄膜を積む。
    これにより等形成膜モードでも平坦界面モードと同等の計算が可能になる。
    Returns: [(h_lo, h_hi, material, fill_fraction, alignment), ...]
    """
    bands, h = [], 0.0
    if col['subst_raise'] > 1e-9:
        bands.append((h, h + col['subst_raise'], SUBST_MATERIAL, 1.0, 'Left'))
        h += col['subst_raise']
    # flat_align: 非隆起列に ENV ギャップを追加して全列の薄膜開始高さを揃える
    if flat_align_height > h + 1e-9:
        bands.append((h, flat_align_height, ENV_MATERIAL, 1.0, 'Left'))
        h = flat_align_height
    for fl in col['film_layers']:
        ftype = fl.get('type', 'simple')
        if ftype == 'simple':
            th = fl.get('thickness', 0.0)
            if th > 1e-9:
                bands.append((h, h + th, fl['material'], 1.0, 'Left'))
                h += th
        elif ftype == 'pair':
            n    = max(1, fl.get('n_pairs', 1))
            th_a = fl.get('th_a', 0.0)
            th_b = fl.get('th_b', 0.0)
            ma   = fl.get('mat_a', '')
            mb   = fl.get('mat_b', '')
            for _ in range(n):
                # B層を基板側（先）、A層を空気側（後）に積む（ppap.py の "Layer A (top)" と同一規約）
                if th_b > 1e-9 and mb:
                    bands.append((h, h + th_b, mb, 1.0, 'Left')); h += th_b
                if th_a > 1e-9 and ma:
                    bands.append((h, h + th_a, ma, 1.0, 'Left')); h += th_a
        elif ftype == 'taper':
            th_total = fl.get('thickness', 0.0)
            n_sl  = max(1, fl.get('n_slices', 10))
            align = fl.get('alignment', 'Center')
            mat   = fl.get('material', '')
            fb    = max(0.0, min(1.0, fl.get('frac_bottom', 1.0)))
            ft    = max(0.0, min(1.0, fl.get('frac_top',    0.5)))
            th_sl = th_total / n_sl
            if th_sl > 1e-9 and mat:
                for i in range(n_sl):
                    frac = fb + (ft - fb) * i / n_sl   # i=0 が下端
                    bands.append((h, h + th_sl, mat, frac, align))
                    h += th_sl
    return bands


def _col_total_height(col):
    bands = _expand_to_bands(col, flat_align_height=0.0)
    return bands[-1][1] if bands else 0.0


def columns_to_layers(columns, flat_align=False):
    """列定義（幅・基板上げ・フィルム積層）を RCWA 水平層リスト（Top-first）に変換する。
    flat_align=True のとき、全列の薄膜開始高さを最大 subst_raise に揃える（平坦界面と同等）。
    """
    total_width = sum(c['width'] for c in columns)
    if total_width <= 1e-9:
        return []

    align_h = max((c.get('subst_raise', 0.0) for c in columns), default=0.0) if flat_align else 0.0
    col_bands = [_expand_to_bands(col, flat_align_height=align_h) for col in columns]

    heights = set([0.0])
    for bands in col_bands:
        for h_lo, h_hi, *_ in bands:
            heights.add(h_lo); heights.add(h_hi)
    heights = sorted(heights)

    result = []
    for i in range(len(heights) - 1):
        h_lo, h_hi = heights[i], heights[i + 1]
        h_mid = (h_lo + h_hi) / 2.0
        segs = []
        for col, bands in zip(columns, col_bands):
            col_frac = col['width'] / total_width
            mat, fill_frac, align = ENV_MATERIAL, 1.0, 'Left'
            for (blo, bhi, bmat, bff, bal) in bands:
                if blo <= h_mid < bhi:
                    mat, fill_frac, align = bmat, bff, bal; break
            _add_col_segs(segs, mat, fill_frac, align, col_frac)
        result.append({'thickness': h_hi - h_lo, 'segments': segs})

    result.reverse()   # Top-first
    return result


def regions_to_layers_2d(regions, field_film_layers):
    """2次元周期アレイ版の columns_to_layers()。列幅の代わりに、各「領域」が
    独立した基板上げ高さ(raise_nm)を持つ円/矩形の2D領域として定義される
    （1Dの列に相当）。成膜レシピ(field_film_layers)は全領域・フィールド
    （領域の外側＝基板上げ0の暗黙領域）で共通——`_expand_to_bands()`は
    無改造で再利用する（列幅を一切見ない関数なので、2Dにもそのまま使える）。

    regions: [{'shape': 'circle', 'cx_nm','cy_nm','radius_nm','raise_nm'}, ...]
             または [{'shape': 'rect', 'cx_nm','cy_nm','hw_x_nm','hw_y_nm','raise_nm'}, ...]
             （'shape'省略時は'circle'扱い）。
    field_film_layers: 1Dの`col['film_layers']`と同じ形式。'simple'/'pair'
    のみを想定（'taper'は列幅方向の先細り概念で2D領域には直接対応しないため、
    呼び出し側で除外すること——taperが混入していても例外にはならないが、
    _expand_to_bandsが返すfill_fraction<1.0は本関数では無視されるため、
    taper由来の部分充填は正しく表現されない）。

    戻り値: `rcwa_s4_array2d_backend.build_array_geometry()`がそのまま
    受け取れる形式のlayersリスト（top-first）。各要素は
    {'thickness_nm','bg_material','disks':[...],'rects':[...]}。

    重なり合う領域の扱いは未定義（領域同士が同じ(x,y)位置で重ならないことを
    呼び出し側のUIで保証すること——重なった場合、同じ高さバンドに複数の
    disks/rects領域が同時に追加され、どちらが上に描画されるかはS4側の
    描画順序に委ねられる）。
    """
    field_bands = _expand_to_bands({'subst_raise': 0.0, 'film_layers': field_film_layers})
    region_bands = [
        _expand_to_bands({'subst_raise': r.get('raise_nm', 0.0), 'film_layers': field_film_layers})
        for r in regions
    ]

    heights = set([0.0])
    for bands in [field_bands] + region_bands:
        for h_lo, h_hi, *_ in bands:
            heights.add(h_lo); heights.add(h_hi)
    heights = sorted(heights)

    def _mat_at(bands, h_mid):
        for (blo, bhi, bmat, *_rest) in bands:
            if blo <= h_mid < bhi:
                return bmat
        return ENV_MATERIAL

    result = []
    for i in range(len(heights) - 1):
        h_lo, h_hi = heights[i], heights[i + 1]
        h_mid = (h_lo + h_hi) / 2.0
        bg_mat = _mat_at(field_bands, h_mid)
        disks, rects = [], []
        for region, bands in zip(regions, region_bands):
            mat = _mat_at(bands, h_mid)
            if region.get('shape', 'circle') == 'rect':
                rects.append({
                    'material': mat,
                    'cx_nm': region.get('cx_nm', 0.0), 'cy_nm': region.get('cy_nm', 0.0),
                    'hw_x_nm': region.get('hw_x_nm', 0.0), 'hw_y_nm': region.get('hw_y_nm', 0.0),
                })
            else:
                disks.append({
                    'material': mat,
                    'cx_nm': region.get('cx_nm', 0.0), 'cy_nm': region.get('cy_nm', 0.0),
                    'radius_nm': region.get('radius_nm', 0.0),
                })
        result.append({'thickness_nm': h_hi - h_lo, 'bg_material': bg_mat,
                        'disks': disks, 'rects': rects})

    result.reverse()   # Top-first
    return result


def _column_edges(columns):
    """列の左から右への絶対[x1, x2)座標(nm)のリストを返す。"""
    edges, x = [], 0.0
    for c in columns:
        edges.append((x, x + c['width'])); x += c['width']
    return edges


def _graded_raise_height(H, x1, x2, s, pitch):
    """位置sにおける、なめらか基板上げ列1本分の高さ寄与(super-Gaussian)。
    h(s) = H * exp(-|((2s-x1-x2)/(x2-x1))|^n), n = 0.02*(x2-x1)
    列の外側でも0にクリップせず裾を計算する(隣接列への侵入を許すため)。
    周期境界(s=0, s=pitch)をまたいだ裾の回り込みも s±pitch の2像で評価する
    (減衰長は列幅程度でpitchよりずっと小さいため、1次像で十分)。
    """
    w = x2 - x1
    if w <= 1e-9 or H <= 1e-9:
        return 0.0
    n = 0.02 * w
    best = 0.0
    for k in (-1, 0, 1):
        u = (2.0 * (s + k * pitch) - x1 - x2) / w
        with np.errstate(over='ignore'):
            val = H * float(np.exp(-np.power(abs(u), n)))
        best = max(best, val)
    return best


def _sharp_raise_height(H, x1, x2, s):
    """従来の矩形段差(列の外側では0、内側では常にH)。"""
    return H if (x1 <= s < x2) else 0.0


def expand_graded_columns(columns, n_slices):
    """'graded'な基板上げを持つ列を、n_slices本の細い子列に展開する前処理。

    各列は自分自身の[x1,x2)幅をn_slices等分する(周期全体を均等分割して
    所属列を検索する方式だと、狭い列がどのビンにも当たらずfilm_layersが
    消失しうるため不採用)。これにより所属は常にその列自身に一意に決まり、
    film_layersが失われることはなく、幅の合計も厳密に保存される。

    各子列の基板上げ高さは、全列(sharp/graded問わず)の高さ寄与関数を
    その子列の中心位置で評価したときの最大値(max envelope)。'sharp'な
    列も含めて展開するのは、sharp列の基板高さも隣接するgraded列からの
    裾の侵入を受けて0ではなくなりうるため。

    'graded'な列が1つも無い場合はこの関数を呼ばず、columns_to_layers()に
    直接渡すこと(既存構造の計算結果を完全に不変に保つため)。
    """
    edges = _column_edges(columns)
    pitch = edges[-1][1] if edges else 0.0
    if pitch <= 1e-9:
        return list(columns)

    n_slices = max(1, int(n_slices))
    out = []
    for (x1, x2), col in zip(edges, columns):
        sub_w = (x2 - x1) / n_slices
        for j in range(n_slices):
            s_c = x1 + (j + 0.5) * sub_w
            h = 0.0
            for (xk1, xk2), ck in zip(edges, columns):
                Hk = ck.get('subst_raise', 0.0)
                if Hk <= 1e-9:
                    continue
                if ck.get('raise_mode', 'sharp') == 'graded':
                    hk = _graded_raise_height(Hk, xk1, xk2, s_c, pitch)
                else:
                    hk = _sharp_raise_height(Hk, xk1, xk2, s_c)
                if hk > h:
                    h = hk
            out.append({'width': sub_w, 'subst_raise': h, 'film_layers': col['film_layers']})
    return out


def get_layer_tuple(wl, layers, nk_fn_map, n_env_val, nk_subst_fn):
    """R9's top-first `layers` dict list -> rcwa_mh_cpp's substrate-first tuple format."""
    n_env = complex(n_env_val)
    layer_list = [(0, complex(nk_subst_fn(wl)), 0)]
    for L in reversed(layers):
        flat = []
        for mat, w in L['segments']:
            if w <= 1e-9: continue
            if   mat == ENV_MATERIAL:   n = n_env
            elif mat == SUBST_MATERIAL: n = complex(nk_subst_fn(wl))
            else:                       n = complex(nk_fn_map[mat](wl))
            flat.append(n); flat.append(w)
        if not flat: continue
        layer_list.append(tuple([L['thickness']/1000.0] + flat))
    layer_list.append((0, n_env, 0))
    return tuple(layer_list)

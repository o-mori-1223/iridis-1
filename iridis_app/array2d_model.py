"""2Dモードの層構成（UIの array_layers / 基板段差）を、S4 2D backend
（RCWA2D.rcwa_s4_array2d_backend）の layers 形式に変換する純粋関数群。Streamlit 非依存。

UI側の層仕様（top-first のリスト）:
  film   : {'type':'film',   'material', 'thickness_nm'}
  disk   : {'type':'disk',   'material', 'bg_material', 'thickness_nm', 'radius_nm', 'cx_nm', 'cy_nm'}
  sphere : {'type':'sphere', 'material', 'bg_material', 'radius_nm', 'n_slices', 'cx_nm', 'cy_nm'}
  hemisphere : sphere と同じキー（平らな面を下にしたドーム形、高さ = 半径）
  hcylinder : {'type':'hcylinder', 'material', 'bg_material', 'radius_nm', 'length_nm', 'axis'('x'|'y'),
               'infinite'(bool: 軸方向の周期いっぱい＝無限に長い棒), 'n_slices', 'cx_nm', 'cy_nm'}
               横に倒したシリンダー（高さ = 直径）
  pair   : {'type':'pair', 'a': <film|disk>, 'b': <film|disk>, 'n_pairs'}
"""
import functools
import json

import numpy as np

from RCWA2D import curved_multilayer
from RCWA2D.material_spec import resolve_nk_fn
from RCWA2D.rcwa_s4_array2d_backend import hcylinder_rect_layers, hemisphere_disk_layers, sphere_disk_layers
from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL, regions_to_layers_2d

PREVIEW_PALETTE = ['#c0392b', '#2980b9', '#27ae60', '#f39c12', '#8e44ad',
                   '#16a085', '#d35400', '#7f8c8d']


def lattice_vectors_nm(pitch_x_nm, pitch_y_nm, gamma_deg):
    """(a1, a2) を nm の np.array 2本で返す（rcwa_s4_array2d_backend.lattice_vectors_um と同じ規約。
    a1=(pitch_x,0), a2=pitch_y を gamma_deg 回転）。"""
    g = np.radians(gamma_deg)
    return np.array([pitch_x_nm, 0.0]), np.array([pitch_y_nm * np.cos(g), pitch_y_nm * np.sin(g)])


def nearest_neighbor_dist_nm(a1, a2):
    """格子点間の最短距離（円盤/球が隣の周期像と重ならない上限半径 = この値/2）。
    正方・矩形格子なら min(|a1|,|a2|)、六方格子なら |a1|=|a2|=|a1-a2|。"""
    return min(float(np.linalg.norm(v)) for v in (a1, a2, a1 - a2))


def expand_layer_spec(lyr):
    """'pair' 型は A/B を n_pairs 回繰り返した実体(film/disk)のリストに展開する。
    それ以外はそのまま1要素のリスト。"""
    if lyr['type'] != 'pair':
        return [lyr]
    return [lyr[k] for _ in range(int(lyr['n_pairs'])) for k in ('a', 'b')]


PARTICLE_TYPES = ('disk', 'sphere', 'hemisphere', 'hcylinder')


def hcylinder_length_nm(sub, a1, a2):
    """横倒しシリンダーの長さ。infinite なら軸方向の周期（端が隣の周期像とつながる）。
    y 方向の無限長は a2 が y 方向（矩形格子）のときだけ定義できる。"""
    if not sub.get('infinite'):
        return float(sub['length_nm'])
    if sub['axis'] == 'x':
        return float(np.linalg.norm(a1))
    if abs(float(a2[0])) > 1e-6:
        raise ValueError("y 方向に無限に長い棒は、格子の角度 90°（矩形格子）のときだけ作れます")
    return abs(float(a2[1]))


def hcylinder_footprint_nm(sub, a1, a2):
    """真上から見た外形の半幅 (hw_x, hw_y)。"""
    half_len, r = 0.5 * hcylinder_length_nm(sub, a1, a2), float(sub['radius_nm'])
    return (half_len, r) if sub['axis'] == 'x' else (r, half_len)


def hcylinder_overlaps(sub, a1, a2):
    """真上から見た外形（長方形）が隣の周期像と重なるか。同じ大きさの軸並行な長方形どうしは、
    中心のずれ v が |v_x| < 2hw_x かつ |v_y| < 2hw_y のとき重なる（ちょうど接するのは可）。"""
    hx, hy = hcylinder_footprint_nm(sub, a1, a2)
    for i in range(-4, 5):
        for j in range(-4, 5):
            if i == 0 and j == 0:
                continue
            v = i * np.asarray(a1) + j * np.asarray(a2)
            if abs(v[0]) < 2 * hx - 1e-6 and abs(v[1]) < 2 * hy - 1e-6:
                return True
    return False


def layer_thickness_nm(sub):
    if sub['type'] in ('sphere', 'hcylinder'):
        return 2.0 * sub['radius_nm']
    if sub['type'] == 'hemisphere':
        return sub['radius_nm']
    return sub.get('thickness_nm', 0.0)


def _register_nk(name, nk_fn_map):
    """ENV/SUBST（「環境と同じ」「基板と同じ」）は backend が常に nk_env_fn/nk_subst_fn を
    使うので登録不要。"""
    if name not in (ENV_MATERIAL, SUBST_MATERIAL) and name not in nk_fn_map:
        nk_fn_map[name] = resolve_nk_fn(name)


def _append_backend_layers(sub, layers, nk_fn_map, lattice=None):
    """film/disk/sphere/hemisphere/hcylinder 1個分を backend の layers 形式に変換して append する。"""
    if sub['type'] == 'film':
        layers.append({'thickness_nm': float(sub['thickness_nm']),
                       'bg_material': sub['material'], 'disks': []})
        _register_nk(sub['material'], nk_fn_map)
        return
    cx, cy = float(sub.get('cx_nm', 0.0)), float(sub.get('cy_nm', 0.0))
    if sub['type'] == 'disk':
        layers.append({
            'thickness_nm': float(sub['thickness_nm']), 'bg_material': sub['bg_material'],
            'disks': [{'material': sub['material'], 'cx_nm': cx, 'cy_nm': cy,
                       'radius_nm': float(sub['radius_nm'])}],
        })
    elif sub['type'] == 'hcylinder':   # staircase近似（矩形）。原点中心で返すのでずらす
        if sub.get('infinite') and lattice is None:
            raise ValueError("無限に長い棒には格子ベクトルが必要です")
        length = hcylinder_length_nm(sub, *lattice) if lattice is not None else float(sub['length_nm'])
        cyl_layers = hcylinder_rect_layers(float(sub['radius_nm']), length, sub['axis'], sub['material'],
                                           int(sub['n_slices']), bg_material=sub['bg_material'])
        for L in cyl_layers:
            for r in L['rects']:
                r['cx_nm'] += cx
                r['cy_nm'] += cy
        layers.extend(cyl_layers)
    else:  # sphere / hemisphere（staircase近似。常に原点中心で返すので必要ならずらす）
        slicer = sphere_disk_layers if sub['type'] == 'sphere' else hemisphere_disk_layers
        sph_layers = slicer(float(sub['radius_nm']), sub['material'], int(sub['n_slices']),
                            bg_material=sub['bg_material'])
        for L in sph_layers:
            for d in L['disks']:
                d['cx_nm'] += cx
                d['cy_nm'] += cy
        layers.extend(sph_layers)
    _register_nk(sub['material'], nk_fn_map)
    _register_nk(sub['bg_material'], nk_fn_map)


def _recipe_key(recipe):
    return json.dumps([{k: v for k, v in fl.items() if k != 'id'} for fl in recipe], sort_keys=True)


@functools.lru_cache(maxsize=16)
def _lens_layers_cached(spec_json, recipe_json, lattice_key):
    spec = json.loads(spec_json)
    a1, a2 = np.array(lattice_key[:2]), np.array(lattice_key[2:])
    return curved_multilayer.build_layers(
        spec['shape'], spec['base_radius_nm'], spec['height_nm'], json.loads(recipe_json),
        concave=spec['concave'], deposition=spec['deposition'], dz_max_nm=spec['dz_max_nm'],
        cx_nm=spec['cx_nm'], cy_nm=spec['cy_nm'], lattice=(a1, a2),
        disk_radius_nm=spec['disk_radius_nm'] if spec.get('film_extent') == 'disk' else None)


def lens_layers(spec, recipe, a1, a2):
    """レンズ状の凹凸＋形に沿った多層膜 →（layers, info）。同じ入力ならキャッシュを返す
    （凹凸の境界面の計算に約1秒かかり、画面の再実行のたびに作り直さないため）。"""
    key = (float(a1[0]), float(a1[1]), float(a2[0]), float(a2[1]))
    return _lens_layers_cached(json.dumps(spec, sort_keys=True), _recipe_key(recipe), key)


def build_layers(array_layers, step_regions=None, step_film=(), lens=None, lattice=None):
    """UIの層構成 →（backend layers（top-first）, nk_fn_map）。
    step_regions を与えると、基板段差（regions_to_layers_2d）を層構成より基板側に積む。
    lens=(spec, recipe) と lattice=(a1, a2) を与えると、レンズ状の凹凸＋多層膜を同様に積む
    （基板段差とは排他。両方あればレンズを優先）。"""
    layers, nk_fn_map = [], {}
    for lyr in array_layers:
        for sub in expand_layer_spec(lyr):
            _append_backend_layers(sub, layers, nk_fn_map, lattice)
    if lens is not None:
        spec, recipe = lens
        layers.extend(lens_layers(spec, recipe, *lattice)[0])
        step_film = recipe
    elif step_regions:
        layers.extend(regions_to_layers_2d(step_regions, list(step_film)))
    else:
        step_film = ()
    for fl in step_film:
        for name in ((fl['mat_a'], fl['mat_b']) if fl.get('type') == 'pair' else (fl['material'],)):
            _register_nk(name, nk_fn_map)
    return layers, nk_fn_map


_SLICED_TYPES = ('sphere', 'hemisphere', 'hcylinder')


def refined_inputs(array_layers, lens=None):
    """階段近似を2倍細かくした (array_layers, lens) を返す（球・半球・横倒しシリンダーの
    n_slices ×2、レンズ状の凹凸の Δz ÷2）。細かくできるものがなければ None。"""
    def refine(sub):
        return dict(sub, n_slices=int(sub['n_slices']) * 2) if sub['type'] in _SLICED_TYPES else sub

    changed = False
    out = []
    for lyr in array_layers:
        if lyr['type'] == 'pair':
            new = dict(lyr, a=refine(lyr['a']), b=refine(lyr['b']))
        else:
            new = refine(lyr)
        changed |= new != lyr
        out.append(new)
    if lens is not None:
        spec, recipe = lens
        lens = (dict(spec, dz_max_nm=float(spec['dz_max_nm']) / 2.0), recipe)
        changed = True
    return (out, lens) if changed else None


def preview_color_map(array_layers, step_film=()):
    mats = []
    for lyr in array_layers:
        for sub in expand_layer_spec(lyr):
            mats.append(sub['material'])
    for fl in step_film:
        mats += [fl['mat_a'], fl['mat_b']] if fl.get('type') == 'pair' else [fl.get('material')]
    mats = [m for m in dict.fromkeys(mats) if m]
    return {m: PREVIEW_PALETTE[i % len(PREVIEW_PALETTE)] for i, m in enumerate(mats)}


def _sub_label(s):
    mat = '基板' if s['material'] == SUBST_MATERIAL else s['material']
    if s['type'] == 'film':
        return f"{mat}膜{s['thickness_nm']:.0f}nm"
    if s['type'] == 'disk':
        return f"{mat}円盤r{s['radius_nm']:.0f}/h{s['thickness_nm']:.0f}"
    if s['type'] == 'hemisphere':
        return f"{mat}半球r{s['radius_nm']:.0f}"
    if s['type'] == 'hcylinder':
        length = '∞' if s.get('infinite') else f"{s['length_nm']:.0f}"
        return f"{mat}横倒しシリンダー{s['axis']}軸r{s['radius_nm']:.0f}/L{length}"
    return f"{mat}球r{s['radius_nm']:.0f}"


def stack_label(array_layers):
    parts = [f"[{_sub_label(l['a'])}+{_sub_label(l['b'])}]×{int(l['n_pairs'])}" if l['type'] == 'pair'
             else _sub_label(l) for l in array_layers]
    return " + ".join(parts) if parts else "(層なし)"

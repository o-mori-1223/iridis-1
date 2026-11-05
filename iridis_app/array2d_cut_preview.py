"""円柱で切り抜いた構造の3Dモデル（S4 に渡す層をそのまま立体にする。S4 計算はしない）。

周期領域と円柱の切り抜き（RCWA2D/cylinder_cut.py）のあとの layers を、層ごとに
円・矩形・多角形を厚さ方向に押し出した柱（角柱）として描く。計算する形そのものなので、
切り抜く前の形に円柱の壁を重ねる 3D 表示（array2d_preview.structure_preview_fig）と違い、
円柱の外側（または穴）は本当に取り除かれている。

- 他の形を中に含む形（切り抜いた膜の円柱が中の台を含む、など）は半透明、それ以外は不透明。
- 環境（空気）は描かない。層の背景は、環境でなければ周期領域いっぱいの半透明の柱で描く。
- 同じ材質・同じ透明度の形は 1 つのメッシュにまとめる（3×3 で形が多くても描画が重くならない）。
- cut_model_obj で、同じ形を Wavefront OBJ（材質ごとのグループ、単位 nm）として書き出せる。
"""
import numpy as np
import plotly.graph_objects as go

from RCWA2D.cylinder_cut import circle_polygon
from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL

from .array2d_preview import MESH_LIGHTING, _AXIS_COLOR, _SUBST_COLOR, _Edges, _axis

_CONTAINER_OPACITY = 0.35   # 中に別の形を含む形
_BG_OPACITY = 0.18          # 層の背景（周期領域いっぱい）


def _shape_outline(kind, s, tol_nm=0.5):
    """形の輪郭（反時計回りの頂点）。円は描画用なので少し粗く（弦と円弧のずれ ≤ 0.5 nm）。"""
    if kind == 'disk':
        return circle_polygon(s['cx_nm'], s['cy_nm'], s['radius_nm'], tol_nm, min_n=48)
    if kind == 'rect':
        cx, cy, hx, hy = s['cx_nm'], s['cy_nm'], s['hw_x_nm'], s['hw_y_nm']
        return [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
    vs = [tuple(v) for v in s['vertices_nm']]
    area = 0.5 * sum(vs[i][0] * vs[(i + 1) % len(vs)][1] - vs[(i + 1) % len(vs)][0] * vs[i][1] for i in range(len(vs)))
    return vs if area >= 0 else vs[::-1]


def _bbox(vs):
    xs, ys = [v[0] for v in vs], [v[1] for v in vs]
    return min(xs), max(xs), min(ys), max(ys)


def _contains_bbox(outer, inner):
    return outer[0] <= inner[0] + 1e-6 and outer[1] >= inner[1] - 1e-6 and \
        outer[2] <= inner[2] + 1e-6 and outer[3] >= inner[3] - 1e-6


def _prism(vs, z0, z1):
    """多角形 vs を z0〜z1 に押し出した柱の頂点と三角形（側面＋上下面。上下面は重心からの扇形）。"""
    n = len(vs)
    xs = [v[0] for v in vs] * 2 + [sum(v[0] for v in vs) / n] * 2
    ys = [v[1] for v in vs] * 2 + [sum(v[1] for v in vs) / n] * 2
    zs = [z0] * n + [z1] * n + [z0, z1]
    cb, ct = 2 * n, 2 * n + 1
    tris = []
    for a in range(n):
        b = (a + 1) % n
        tris += [(a, b, n + b), (a, n + b, n + a), (cb, b, a), (ct, n + a, n + b)]
    return np.column_stack([xs, ys, zs]), np.array(tris)


class _MeshGroups:
    """（材質, 透明度）ごとに頂点と三角形をまとめる。"""

    def __init__(self):
        self.groups = {}

    def add(self, material, opacity, verts, tris):
        g = self.groups.setdefault((material, opacity), {'v': [], 't': [], 'n': 0})
        g['v'].append(verts)
        g['t'].append(tris + g['n'])
        g['n'] += len(verts)

    def items(self):
        for (mat, op), g in self.groups.items():
            yield mat, op, np.vstack(g['v']), np.vstack(g['t'])


def build_cut_model(layers, A1, A2, z_bottom_nm=0.0, subst_h_nm=None, draw_substrate=True):
    """切り抜き後の layers（top-first）→（_MeshGroups, _Edges, 上端の高さ, 基板の厚さ）。"""
    A1, A2 = np.asarray(A1, float), np.asarray(A2, float)
    foot = [tuple(s * A1 / 2 + t * A2 / 2) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    total = sum(L['thickness_nm'] for L in layers)
    subst_h = subst_h_nm if subst_h_nm is not None else max(0.35 * total, 40.0)
    groups, edges = _MeshGroups(), _Edges()
    if draw_substrate:
        groups.add(SUBST_MATERIAL, 1.0, *_prism(foot, z_bottom_nm - subst_h, z_bottom_nm))
    z = z_bottom_nm
    for L in reversed(layers):   # 下の層から積む
        z0, z1 = z, z + L['thickness_nm']
        z = z1
        if L['bg_material'] != ENV_MATERIAL:
            groups.add(L['bg_material'], _BG_OPACITY, *_prism(foot, z0, z1))
        shapes = ([('disk', d) for d in L.get('disks', []) if d['radius_nm'] > 0] +
                  [('rect', r) for r in L.get('rects', []) if r['hw_x_nm'] > 0 and r['hw_y_nm'] > 0] +
                  [('poly', p) for p in L.get('polygons', []) if len(p['vertices_nm']) >= 3])
        outlines = [(s['material'], _shape_outline(k, s)) for k, s in shapes]
        boxes = [_bbox(vs) for _, vs in outlines]
        for i, (mat, vs) in enumerate(outlines):
            if mat == ENV_MATERIAL:   # 空気の領域（穴など）は描かない。縁だけ線で示す
                edges.polyline([v[0] for v in vs] + [vs[0][0]], [v[1] for v in vs] + [vs[0][1]], [z1] * (len(vs) + 1))
                continue
            container = any(j != i and _contains_bbox(boxes[i], boxes[j]) and boxes[i] != boxes[j]
                            for j in range(len(outlines)))
            groups.add(mat, _CONTAINER_OPACITY if container else 1.0, *_prism(vs, z0, z1))
            edges.polyline([v[0] for v in vs] + [vs[0][0]], [v[1] for v in vs] + [vs[0][1]], [z1] * (len(vs) + 1))
    return groups, edges, z, subst_h


def cut_model_fig(layers, A1, A2, color_map, subst_label, z_bottom_nm=0.0, title=""):
    """切り抜き後の layers の 3D モデル（plotly Figure）。"""
    groups, edges, top, subst_h = build_cut_model(layers, A1, A2, z_bottom_nm)

    def label(m):
        if m == SUBST_MATERIAL:
            return f"基板 ({subst_label})"
        return "環境と同じ" if m == ENV_MATERIAL else m

    traces = []
    for mat, op, v, t in groups.items():
        color = _SUBST_COLOR if mat == SUBST_MATERIAL else color_map.get(mat, "#7f8c8d")
        traces.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=t[:, 0], j=t[:, 1], k=t[:, 2],
                                color=color, opacity=op, flatshading=True, lighting=MESH_LIGHTING,
                                hoverinfo='text', hovertext=label(mat) + ("（中に別の形を含む・半透明）" if op < 1 else ""),
                                showscale=False))
    traces.append(edges.trace())
    A1, A2 = np.asarray(A1, float), np.asarray(A2, float)
    foot = [s * A1 / 2 + t * A2 / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
    z_lo = z_bottom_nm - subst_h
    traces.append(go.Scatter3d(x=[c[0] for c in foot], y=[c[1] for c in foot], z=[z_lo] * 5, mode='lines',
                               line=dict(color='black', width=3), hoverinfo='skip', showlegend=False))
    fig = go.Figure(data=traces)
    fig.update_layout(
        height=560, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white", font=dict(color=_AXIS_COLOR),
        scene=dict(xaxis=_axis("x [nm]"), yaxis=_axis("y [nm]"), zaxis=_axis("z [nm]"),
                   aspectmode="data", camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))),
        title=dict(text=title or "切り抜いた形の 3D モデル（S4 に渡す形そのまま）", font=dict(size=13)))
    return fig


def cut_model_obj(layers, A1, A2, subst_label, z_bottom_nm=0.0, include_substrate=False):
    """同じ形を Wavefront OBJ（文字列）にする。材質ごとのグループ、単位 nm。基板は既定で含めない
    （周期領域いっぱいの板なので、切り抜いた形だけを取り出したいときに邪魔になる）。"""
    groups, _, _, _ = build_cut_model(layers, A1, A2, z_bottom_nm, draw_substrate=include_substrate)
    lines = ["# Iridis: 円柱で切り抜いた構造（S4 に渡す形）。単位 nm。", "# 半透明で描く形（中に別の形を含む・層の背景）も含む"]
    base = 1
    for mat, op, v, t in groups.items():
        name = f"substrate_{subst_label}" if mat == SUBST_MATERIAL else mat
        name = (name + ("_container" if op < 1 else "")).replace(" ", "_")   # OBJ のグループ名に空白は使えない
        lines.append(f"g {name}")
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v]
        lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in t]
        base += len(v)
    return "\n".join(lines) + "\n"

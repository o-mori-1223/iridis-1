"""1 次元の列構造（基板に段差をつけ、その上に 2 材料の多層膜を段差に沿って積んだもの）の逆設計。

周期は 2 つの列（段差の上・下）からなる。探す寸法は 2 つの列の幅・段差の高さ・2 材料の膜厚・ペアの数。
膜のモデルは「🧱 構造」の列ビルダーと同じ（等形成膜。列ごとに膜を積む）。
"""
import numpy as np

from ...structure_builder import columns_to_layers
from ..runner import Level, Var, snap_vars


class SteppedColumns:
    name, kind = 'columns', '1d'
    levels = (Level(15, 0, 2, 41), Level(31, 0, 2, 41), Level(61, 0, 1, 81))

    def __init__(self, pair, width=(100.0, 400.0), raise_=(20.0, 250.0), thickness=(30.0, 150.0), n_pairs=(3, 5), step=1.0):
        """pair: (材料 1, 材料 2)。どちらを上にするかも探す。"""
        self.pair, self.materials, self.label = tuple(pair), tuple(pair), '・'.join(pair) + ' の列構造'
        self.variables = [Var('w1', '段差の上の列の幅', *map(float, width), step), Var('w2', '段差の下の列の幅', *map(float, width), step),
                          Var('h', '段差の高さ', *map(float, raise_), step),
                          Var('ta', '上の材料の膜厚', *map(float, thickness), step), Var('tb', '下の材料の膜厚', *map(float, thickness), step)]
        if n_pairs[1] > n_pairs[0]:
            self.variables.append(Var('n', 'ペアの数', n_pairs[0], n_pairs[1], 1.0, ''))
        self.variables.append(Var('order', '材料の順', 0, 1, 1.0, ''))
        self.n_fixed = int(n_pairs[0])

    def project(self, X):
        return snap_vars(X, self.variables)

    def _values(self, x):
        d = {v.key: float(t) for v, t in zip(self.variables, x)}
        top, bottom = self.pair if d['order'] < 0.5 else self.pair[::-1]
        return d, top, bottom, int(round(d.get('n', self.n_fixed)))

    def _columns(self, x):
        d, top, bottom, n = self._values(x)
        film = {'type': 'pair', 'mat_a': top, 'th_a': d['ta'], 'mat_b': bottom, 'th_b': d['tb'], 'n_pairs': n}
        return [{'width': d['w1'], 'subst_raise': d['h'], 'raise_mode': 'sharp', 'film_layers': [dict(film)]},
                {'width': d['w2'], 'subst_raise': 0.0, 'raise_mode': 'sharp', 'film_layers': [dict(film)]}]

    def geometry(self, x, level):
        c = self._columns(x)
        return {'pitch_nm': c[0]['width'] + c[1]['width'], 'layers': columns_to_layers(c)}

    def describe(self, x):
        d, top, bottom, n = self._values(x)
        return {'summary': f"列の幅 {d['w1']:g}（段差の上）/{d['w2']:g}・段差 {d['h']:g} nm、"
                           f"{top} {d['ta']:g}（上）/{bottom} {d['tb']:g} nm × {n} ペア",
                'dims': [['周期', d['w1'] + d['w2']], ['段差の上の列の幅', d['w1']], ['段差の下の列の幅', d['w2']],
                         ['段差の高さ', d['h']], [f'{top}（ペアの上側）', d['ta']], [f'{bottom}（ペアの下側）', d['tb']], ['ペアの数', n]],
                'apply': {'mode': '1d', 'columns': self._columns(x)}}

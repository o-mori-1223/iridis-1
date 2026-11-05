"""幅1050nm(上げ無し)+幅350nm(275nm上げ)、周期1400nmの2列構造をGsolverの
テキストインポート用に書き出す。350/1400=25%（ユーザー提供画像の「基盤上げの
割合: 25%」と一致）。

膜厚: TiO2=60nm, SiO2=95nm, 7.5ペア（基板側・空気側ともTiO2、他の構造と同じ
"simple TiO2 1層 + (SiO2,TiO2)ペア x7" の構成）。基板はSUS(n=1.71+2.88j)。
TiO2/SiO2の数値安定化用微小損失(k=1e-4)も他の構造と同一。

実行するとGsolverインポート用テキスト(data/gsolver_import_2col_1050_350.txt)と、
突き合わせ用のrcwa_mh参照スペクトル(data/rcwa_mh_2col_1050_350_result.json、
法線入射・s/p・380-780nm 10nm刻み・NORDER=43)を書き出す。

Gsolver側の手順: Parameters: Units=Microns, Period=1.4。ORDERS=43以上に設定。
File > Import Text で data/gsolver_import_2col_1050_350.txt を読み込み
（Approximateは押さない）、Listing/RUNで直接RUN。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/<分類>/ から見たリポジトリ直下
sys.path.insert(0, _ROOT)
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from RCWA2D.gsolver_export import write_gsolver_import

TH_A, TH_B = 60.0, 95.0          # TiO2, SiO2
N_PAIRS = 7                       # + bottom TiO2 half-pair = 7.5 pairs total
COL_WS = [1050.0, 350.0]
RAISE_H = 275.0
RAISES = [0.0, RAISE_H]           # 幅350nmの列のみ上げ
PITCH_NM = sum(COL_WS)            # 1400.0

N_TIO2 = 2.30 + 1e-4j
N_SIO2 = 1.45 + 1e-4j
N_SUS = 1.71 + 2.88j

NK_FN_MAP = {'TiO2': lambda wl: N_TIO2, 'SiO2': lambda wl: N_SIO2}
NK_SUBST_FN = lambda wl: N_SUS

GLOBAL_FILM = [
    {'type': 'simple', 'material': 'TiO2', 'thickness': TH_A},
    {'type': 'pair', 'mat_a': 'TiO2', 'th_a': TH_A, 'mat_b': 'SiO2', 'th_b': TH_B, 'n_pairs': N_PAIRS},
]


def build_columns():
    return [{'width': w, 'subst_raise': r, 'film_layers': GLOBAL_FILM}
            for w, r in zip(COL_WS, RAISES)]


def build_layers():
    return columns_to_layers(build_columns(), flat_align=False)


if __name__ == '__main__':
    import json

    from RCWA2D import rcwa_mh

    layers = build_layers()
    print(f"pitch = {PITCH_NM} nm, {len(COL_WS)} columns, {len(layers)} RCWA layers")
    for w, r in zip(COL_WS, RAISES):
        print(f"  width={w:>6.1f}nm  raise={r:>6.1f}nm")

    out_txt = os.path.join(_ROOT, 'data', 'gsolver_import_2col_1050_350.txt')
    write_gsolver_import(layers, out_txt)
    print(f"wrote: {out_txt}")

    NORDER = 43
    p0 = NORDER // 2
    PITCH_UM = PITCH_NM / 1000.0
    results = {'s': {}, 'p': {}}
    for pol in ('s', 'p'):
        for wl in range(380, 781, 10):
            lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
            ir, it = rcwa_mh.Rcwa1d(pol, wl / 1000.0, 0.0, PITCH_UM, lt, NORDER)
            results[pol][wl] = float(ir[p0])

    out_json = os.path.join(_ROOT, 'data', 'rcwa_mh_2col_1050_350_result.json')
    with open(out_json, 'w') as f:
        json.dump(results, f)
    print(f"wrote: {out_json}")

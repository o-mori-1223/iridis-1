"""スウォッチ・角度チャート・スペクトルチャートを、参考ツール
`Gsolver'S COLOR.html`（プロジェクトルート直下）と見た目・機能を完全に統一した
生HTML/JSで描画する。ロジックはそのHTMLファイルからほぼ検証済みの移植
（`color_display_template.html`参照。差分はそのファイル内のDEVIATIONコメント3箇所のみ）。

Streamlitのst.color_picker/Plotlyは寸法・スタイルが固定でHTML版と揃えられない
ため、st.iframe()で完全に独立したiframeとして埋め込む（2026-11-05 に、非推奨になった
st.components.v1.html() から置き換え。高さは従来どおり固定）。
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st

_TEMPLATE_PATH = Path(__file__).parent / 'color_display_template.html'
_PLACEHOLDER = '__KOUMYOU_INJECTED_JSON__'

_HEIGHT_WITH_SWATCHES = 2050
_HEIGHT_WITHOUT_SWATCHES = 1800


def fill_nan(values, lambdas):
    """[n_angle, n_wl] の配列の NaN を、同じ角度の行の中で波長方向に線形補間して埋める
    （端は最も近い有限値で外挿）。行が全て NaN なら、角度方向に同じ処理をする。
    それでも残る NaN（全点 NaN）は 0。戻り値は (埋めた配列, NaNだった位置の bool 配列)。
    JSON には NaN を入れられない（ブラウザ側の JSON.parse が失敗して表示ごと消える）ため、
    表示に渡す前に必ず通す。"""
    a = np.array(values, dtype=float, ndmin=2)
    mask = ~np.isfinite(a)
    if not mask.any():
        return a, mask
    x = np.asarray(lambdas, dtype=float)
    for row, m in zip(a, mask):
        if m.any() and not m.all():
            row[m] = np.interp(x[m], x[~m], row[~m])
    still = ~np.isfinite(a)
    if still.any():
        idx = np.arange(a.shape[0], dtype=float)
        for col in a.T:
            m = ~np.isfinite(col)
            if m.any() and not m.all():
                col[m] = np.interp(idx[m], idx[~m], col[~m])
    a[~np.isfinite(a)] = 0.0
    return a, mask


def render_gsolver_style_display(lambdas, thetas, values,
                                  show_swatches, swatch_thetas, swatch_values,
                                  target_wavelengths=None, height=None):
    """スウォッチ(0°/30°/60°)＋角度チャート＋スペクトルチャートを1つの
    埋め込みHTMLブロックとして描画する。

    lambdas, thetas: 1次元配列(nm, deg)。
    values: [len(thetas)][len(lambdas)] の2次元配列（角度掃引の反射率）。
    show_swatches: Falseならスウォッチカードごと非表示にする
        （既存コードの `wl_option == spMenu[0]` ゲートに対応）。
    swatch_thetas, swatch_values: 掃引とは独立に0°/30°/60°で個別計算した
        反射率（掃引の角度刻みが30°・60°ちょうどに当たらない場合でも
        スウォッチが「未入力」にならないようにするため。掃引matrixを
        そのまま流用しないこと）。
    target_wavelengths: 角度チャートに表示する波長(nm)のリスト。
        Noneなら`Gsolver'S COLOR.html`と同じ固定5本(400/470/540/600/700)。
    height: iframeの高さ(px)。旧 st.components.v1.html には自動リサイズが
        無かったため固定値にしていた（st.iframe には height="content" があるが、
        見た目を確かめるまでは従来の固定値のまま）。
        Noneならshow_swatchesの有無に応じたデフォルト値を使う。

    NaN を含む場合は fill_nan() で補間してから描画する。戻り値は補間した点の数
    （values と swatch_values の合計。呼び出し側で警告表示に使う）。
    """
    template = _TEMPLATE_PATH.read_text(encoding='utf-8')

    values, m1 = fill_nan(values, lambdas)
    if len(swatch_values):
        swatch_values, m2 = fill_nan(swatch_values, lambdas)
    else:
        swatch_values, m2 = [], np.zeros(0, dtype=bool)

    payload = {
        'lambdas': [float(v) for v in lambdas],
        'thetas': [float(v) for v in thetas],
        'values': [[float(v) for v in row] for row in values],
        'showSwatches': bool(show_swatches),
        'swatchThetas': [float(v) for v in swatch_thetas],
        'swatchValues': [[float(v) for v in row] for row in swatch_values],
    }
    if target_wavelengths:
        payload['targetWavelengthsAngleChart'] = [float(v) for v in target_wavelengths]

    html = template.replace(_PLACEHOLDER, json.dumps(payload, allow_nan=False))

    if height is None:
        height = _HEIGHT_WITH_SWATCHES if show_swatches else _HEIGHT_WITHOUT_SWATCHES

    st.iframe(html, height=height)   # 自前のテンプレートから作った HTML（外から来た文字列は入れない）
    return int(m1.sum() + m2.sum())

"""1Dモードの構造図（Film Stack Visualization）・周期構造断面図・層テーブル。"""
import pandas as pd
import plotly.graph_objects as go

from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL, _expand_to_bands


def create_structure_figure(layers, pitch_nm, nk_subst_name, col_widths=None):
    shapes, annotations = [], []
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
              '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    uniq_mat = sorted({m for L in layers for m, w in L['segments']
                       if m not in (ENV_MATERIAL, SUBST_MATERIAL)})
    color_map   = {m: colors[i % len(colors)] for i, m in enumerate(uniq_mat)}
    env_color   = 'rgba(240,240,240,0.95)'
    subst_color = 'rgba(160,200,230,0.9)'

    total_th  = sum(L['thickness'] for L in layers)
    subst_th  = total_th * 0.2 if total_th > 0 else 50

    shapes.append(go.layout.Shape(
        type="rect", xref="x", yref="y",
        x0=0, y0=-subst_th, x1=pitch_nm, y1=0,
        fillcolor="lightgrey", line_width=0
    ))
    annotations.append(dict(
        x=pitch_nm/2, y=-subst_th/2,
        text=nk_subst_name, showarrow=False, font=dict(color="black")
    ))

    current_y = 0.0
    for L in reversed(layers):
        d, cursor = L['thickness'], 0.0
        for mat, w in L['segments']:
            x0, x1 = pitch_nm * cursor, pitch_nm * (cursor + w)
            if   mat == ENV_MATERIAL:   clr = env_color
            elif mat == SUBST_MATERIAL: clr = subst_color
            else:                       clr = color_map.get(mat, '#cccccc')
            shapes.append(go.layout.Shape(
                type="rect", xref="x", yref="y",
                x0=x0, y0=current_y, x1=x1, y1=current_y + d,
                fillcolor=clr, line=dict(color="black", width=1)
            ))
            if mat not in (ENV_MATERIAL, SUBST_MATERIAL) and (x1-x0) > pitch_nm*0.04:
                annotations.append(dict(
                    x=(x0+x1)/2, y=current_y+d/2,
                    text=f"{mat}<br>{d:g}nm", showarrow=False,
                    font=dict(color="white", size=10)
                ))
            elif mat == SUBST_MATERIAL:
                annotations.append(dict(
                    x=(x0+x1)/2, y=current_y+d/2,
                    text=f"基板<br>{d:g}nm", showarrow=False,
                    font=dict(color="black", size=9)
                ))
            cursor += w
        current_y += d

    # 列境界を点線で表示
    if col_widths:
        tw = sum(col_widths)
        cx = 0.0
        for cw in col_widths[:-1]:
            cx += cw
            xp = pitch_nm * cx / tw
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y",
                x0=xp, y0=-subst_th, x1=xp, y1=current_y,
                line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot")
            ))

    fig = go.Figure()
    fig.update_layout(
        title="Film Stack Visualization",
        xaxis_title="Position (nm)", yaxis_title="Height (nm)",
        xaxis=dict(range=[0, pitch_nm], showgrid=False, zeroline=False),
        yaxis=dict(range=[-subst_th, current_y*1.1 if current_y>0 else 100],
                   scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False),
        shapes=shapes, annotations=annotations, plot_bgcolor='white'
    )
    return fig


def get_structure_breakdown_df(columns, flat_align=False):
    """各高さでの列材料を示す DataFrame を返す。"""
    align_h = max((c.get('subst_raise', 0.0) for c in columns), default=0.0) if flat_align else 0.0
    col_bands = [_expand_to_bands(col, flat_align_height=align_h) for col in columns]
    heights = set([0.0])
    for bands in col_bands:
        for h_lo, h_hi, *_ in bands:
            heights.add(h_lo); heights.add(h_hi)
    heights = sorted(heights)
    data = []
    for i in range(len(heights) - 1):
        h_lo, h_hi = heights[i], heights[i + 1]
        h_mid = (h_lo + h_hi) / 2
        row = {'高さ [nm]': f'{h_lo:.1f}〜{h_hi:.1f}', '厚さ [nm]': round(h_hi - h_lo, 1)}
        unique_mats = set()
        for ci, (col, bands) in enumerate(zip(columns, col_bands)):
            mat = ENV_MATERIAL
            for blo, bhi, bmat, *_ in bands:
                if blo <= h_mid < bhi:
                    mat = bmat; break
            label = '空気' if mat == ENV_MATERIAL else ('基板' if mat == SUBST_MATERIAL else mat)
            _tag = ', graded近似' if col.get('raise_mode') == 'graded' else ''
            row[f'C{ci+1}({col["width"]:.0f}nm{_tag})'] = label
            unique_mats.add(mat)
        n_uniq = len(unique_mats)
        row['層の種類'] = '均一層' if n_uniq == 1 else f'{n_uniq}材料 格子層'
        data.append(row)
    return pd.DataFrame(data)


def create_crosssection_fig(layers, pitch_nm, col_widths=None, n_repeats=2):
    """2D 周期構造断面図（複数周期表示）を生成する。"""
    shapes, annotations = [], []
    palette = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    uniq_mat = sorted({m for L in layers for m, w in L['segments']
                       if m not in (ENV_MATERIAL, SUBST_MATERIAL)})
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(uniq_mat)}
    env_color   = 'rgba(210,210,210,0.30)'
    subst_color = 'rgba(120,160,210,0.85)'
    subst_bg    = 'rgba(160,160,160,0.70)'

    total_h  = sum(L['thickness'] for L in layers)
    subst_sh = max(total_h * 0.10, 40.0)
    total_xw = pitch_nm * n_repeats

    for rep in range(n_repeats):
        xoff = pitch_nm * rep
        # 基板ブロック
        shapes.append(go.layout.Shape(
            type='rect', xref='x', yref='y',
            x0=xoff, y0=-subst_sh, x1=xoff + pitch_nm, y1=0,
            fillcolor=subst_bg, line=dict(color='grey', width=0.5)
        ))
        current_y = 0.0
        for L in reversed(layers):
            d = L['thickness']
            cursor = 0.0
            for mat, w in L['segments']:
                if w < 1e-9:
                    continue
                x0 = xoff + pitch_nm * cursor
                x1 = xoff + pitch_nm * (cursor + w)
                clr = (env_color   if mat == ENV_MATERIAL   else
                       subst_color if mat == SUBST_MATERIAL else
                       color_map.get(mat, '#cccccc'))
                shapes.append(go.layout.Shape(
                    type='rect', xref='x', yref='y',
                    x0=x0, y0=current_y, x1=x1, y1=current_y + d,
                    fillcolor=clr, line=dict(color='rgba(0,0,0,0.20)', width=0.5)
                ))
                # 材料名ラベル（最初の周期のみ、幅が十分なとき）
                if rep == 0 and mat not in (ENV_MATERIAL, SUBST_MATERIAL):
                    if (x1 - x0) > pitch_nm * 0.04 and d > 5:
                        annotations.append(dict(
                            x=(x0 + x1) / 2, y=current_y + d / 2,
                            text=mat, showarrow=False,
                            font=dict(size=8, color='white'), bgcolor='rgba(0,0,0,0)'
                        ))
                cursor += w
            current_y += d

        # 列境界点線 + 列番号ラベル
        if col_widths:
            tw = sum(col_widths)
            cx = 0.0
            for ci, cw in enumerate(col_widths):
                xp = xoff + pitch_nm * cx / tw
                if ci > 0:
                    shapes.append(go.layout.Shape(
                        type='line', xref='x', yref='y',
                        x0=xp, y0=-subst_sh, x1=xp, y1=total_h,
                        line=dict(color='rgba(0,0,0,0.45)', width=1, dash='dot')
                    ))
                if rep == 0:
                    annotations.append(dict(
                        x=xoff + pitch_nm * (cx + cw / 2) / tw,
                        y=-subst_sh * 0.65,
                        text=f'<b>C{ci+1}</b><br>{cw:.0f}',
                        showarrow=False, font=dict(size=9),
                        align='center'
                    ))
                cx += cw

        # 周期境界（実線）
        if rep > 0:
            shapes.append(go.layout.Shape(
                type='line', xref='x', yref='y',
                x0=xoff, y0=-subst_sh, x1=xoff, y1=total_h,
                line=dict(color='black', width=1.5)
            ))

    # 右端境界線
    shapes.append(go.layout.Shape(
        type='line', xref='x', yref='y',
        x0=total_xw, y0=-subst_sh, x1=total_xw, y1=total_h,
        line=dict(color='black', width=1.5)
    ))

    # 凡例
    legend_shapes, legend_ann = [], []
    ly = total_h * 0.98
    lx0 = total_xw * 1.01
    lx1 = lx0 + pitch_nm * 0.06
    for mat, clr in color_map.items():
        legend_shapes.append(go.layout.Shape(
            type='rect', xref='x', yref='y',
            x0=lx0, y0=ly - total_h * 0.04, x1=lx1, y1=ly,
            fillcolor=clr, line_width=1
        ))
        legend_ann.append(dict(
            x=lx1 + pitch_nm * 0.01, y=ly - total_h * 0.02,
            text=mat, showarrow=False, xanchor='left',
            font=dict(size=10)
        ))
        ly -= total_h * 0.07

    fig = go.Figure()
    fig.update_layout(
        title=dict(text='周期構造断面図  ( x: 位置, y: 高さ )', font=dict(size=14)),
        shapes=shapes + legend_shapes,
        annotations=annotations + legend_ann,
        xaxis=dict(title='x 位置 [nm]', range=[-pitch_nm * 0.01, total_xw + pitch_nm * 0.30],
                   showgrid=False, zeroline=False),
        yaxis=dict(title='高さ [nm]', range=[-subst_sh * 1.3, total_h * 1.05],
                   showgrid=True, gridcolor='rgba(200,200,200,0.4)', zeroline=True,
                   zerolinecolor='rgba(0,0,0,0.4)', zerolinewidth=1.5),
        height=max(380, min(720, int(total_h * 0.55 + 120))),
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig

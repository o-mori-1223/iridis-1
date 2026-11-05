"""保存した記録（runs の 1 行）からスペクトルを取り出し、図（plotly / PNG）を作る。Streamlit 非依存。

- angles(row) / spectrum(row, qkey, theta) : 記録から入射角の一覧・1 本のスペクトルを取り出す
- result_figures(row)                       : 結果の図 {名前: plotly Figure}（画面表示と HTML 保存用）
- result_pngs(row)                          : 同じ内容の PNG {ファイル名: bytes}（matplotlib。Chrome 不要）
- figure_files(name, fig)                   : plotly の図 → 保存するファイル（.html と、画面で描き直す用の .json）
"""
import io

import numpy as np
import plotly.graph_objects as go

from .records import from_jsonable_array

QUANTITIES = {
    "2d": {"R0 (0次/鏡面)": "R0", "R_total (全伝搬次数)": "R_total", "T0 (0次)": "T0", "T_total (全伝搬次数)": "T_total"},
    "1d": {"Rp (指定次数)": "Rp", "Rs (指定次数)": "Rs", "Rp 全次数合算": "Rp_total", "Rs 全次数合算": "Rs_total"},
}
# 角度掃引のヒートマップに使う代表の量
MAIN_QUANTITY = {"2d": "R0", "1d": "Rp"}


def angles(row):
    r = row["result"]
    if row["mode"] == "2d":
        return [float(t) for t in r["theta_deg"]]
    return [float(r["inc_angle"])] if r["mode"] == "single" else [float(t) for t in r["angle_ar"]]


def spectrum(row, qkey, theta_deg=None):
    """記録 row から、入射角 theta_deg（None なら最初の角度。無ければ最も近い角度）での
    スペクトルを返す: (波長, 値, 実際の角度)。その量が無い記録なら None。"""
    r = row["result"]
    th = angles(row)
    i = 0 if theta_deg is None else int(np.argmin(np.abs(np.array(th) - theta_deg)))
    wl = from_jsonable_array(r["wl_nm"])
    if row["mode"] == "2d":
        if qkey not in QUANTITIES["2d"].values():
            return None
        z = from_jsonable_array(r[qkey])
        return wl, (z if z.ndim == 1 else z[i]), th[i]
    if qkey not in QUANTITIES["1d"].values():
        return None
    if r["mode"] == "single":
        ir = from_jsonable_array(r["irp" if qkey.startswith("Rp") else "irs"])
        y = np.clip(ir.sum(axis=1), 0, 1) if qkey.endswith("_total") else ir[:, r["norder"] // 2 + r.get("repr_order", 0)]
        return wl, y, th[0]
    key = {"Rp": "Rp2d", "Rs": "Rs2d", "Rp_total": "Rp_total_2d", "Rs_total": "Rs_total_2d"}[qkey]
    return wl, from_jsonable_array(r[key])[i], th[i]


def spectra_fig(series, title, mode="lines+markers"):
    """series: [(名前, 波長, 値)] の折れ線。計算画面と同じく、透過（名前が T で始まる）は点線。"""
    fig = go.Figure()
    for name, wl, y in series:
        fig.add_trace(go.Scatter(x=wl, y=y, name=name, mode=mode,
                                 line=dict(dash="dot") if str(name).startswith("T") else None))
    fig.update_layout(title=title, xaxis_title="Wavelength (nm)", yaxis_title="R, T", yaxis=dict(range=[0, 1.05]),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def heatmap_fig(row, qname, qkey):
    th = angles(row)
    z = np.array([spectrum(row, qkey, t)[1] for t in th])
    wl = from_jsonable_array(row["result"]["wl_nm"])
    fig = go.Figure(go.Heatmap(x=wl, y=th, z=z, colorscale="Viridis", zmin=0, zmax=1, colorbar=dict(title=qkey)))
    fig.update_layout(title=f"{qname} vs 波長・入射角", xaxis_title="Wavelength (nm)",
                      yaxis_title="入射角 θ (deg)", height=420)
    return fig


def result_figures(row):
    """結果の図。単一角度ならスペクトル 1 枚、角度掃引なら代表量のヒートマップと最初の角度のスペクトル。"""
    qmap = QUANTITIES[row["mode"]]
    th = angles(row)
    figs = {"spectrum": spectra_fig([(name, *spectrum(row, q)[:2]) for name, q in qmap.items()],
                                    f"{row['title']} — θ={th[0]:g}°")}
    if len(th) > 1:
        main = MAIN_QUANTITY[row["mode"]]
        qname = next(n for n, q in qmap.items() if q == main)
        figs["heatmap"] = heatmap_fig(row, qname, main)
    return figs


def result_pngs(row):
    """result_figures と同じ内容の PNG（matplotlib で描く。plotly の PNG 出力は Chrome が要るため）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}
    qmap = QUANTITIES[row["mode"]]
    th = angles(row)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for q in qmap.values():   # 凡例は英字の名前（matplotlib の既定フォントに日本語が無い）
        wl, y, _ = spectrum(row, q)
        ax.plot(wl, y, marker="o", ms=2.5, lw=1.4, label=q)
    ax.set(xlabel="Wavelength (nm)", ylabel="R, T", ylim=(0, 1.05), title=f"θ={th[0]:g}°")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out["spectrum.png"] = _png(fig, plt)
    if len(th) > 1:
        main = MAIN_QUANTITY[row["mode"]]
        z = np.array([spectrum(row, main, t)[1] for t in th])
        wl = from_jsonable_array(row["result"]["wl_nm"])
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        im = ax.pcolormesh(wl, th, z, shading="nearest", vmin=0, vmax=1, cmap="viridis")
        fig.colorbar(im, ax=ax, label=main)
        ax.set(xlabel="Wavelength (nm)", ylabel="θ (deg)", title=f"{main} vs wavelength / angle")
        out["heatmap.png"] = _png(fig, plt)
    return out


def _png(fig, plt):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def figure_files(name, fig):
    """plotly の図 → {ファイル名: (bytes, content_type)}。HTML はブラウザで開ける（plotly.js は CDN から読む
    ので小さい）。JSON は画面で描き直す用。kaleido と Chrome が使える PC なら PNG も付ける。"""
    files = {
        f"{name}.html": (fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"), "text/html"),
        f"{name}.json": (fig.to_json().encode("utf-8"), "application/json"),
    }
    try:
        files[f"{name}.png"] = (fig.to_image(format="png", scale=2), "image/png")
    except Exception:   # kaleido / Chrome が無い（WSL では通常こちら）
        pass
    return files

"""保存した計算条件（runs.params）を画面の入力欄（st.session_state のキー）へ書き戻す。

ss は st.session_state（テストでは普通の dict）。ボタンの on_click コールバックの中で呼ぶ
（入力欄が作られる前なので書き換えられる）。戻り値は、そのまま読み込めなかった点の注意書きのリスト。

層・列・上げ領域・成膜レシピは id ごとに入力欄の key（例: al_r_<id>）を持つ。前の入力欄の値が
session_state に残っていると、そちらが優先されて読み込んだ値が無視されるので、読み込むときは
id を新しく振り直す（カウンタ _array_layer_id などは増える一方なので、新しい id は前と重ならない）。
"""
import copy

import numpy as np

from RCWA2D.structure_builder import ENV_MATERIAL, SUBST_MATERIAL

from .records import from_jsonable_array, nk_snapshot

# 「環境と同じ」「基板と同じ」を表す印。nk ファイルではないので、無い材質の確認から外す
_SENTINELS = (ENV_MATERIAL, SUBST_MATERIAL)

# レンズ編集欄の固定 key（値は lens_spec から入るので、残っている値を消しておく）
_LENS_KEYS = ("lens_extent", "lens_disk_r", "lens_shape", "lens_concave", "lens_dep", "lens_a", "lens_h",
              "lens_dz", "lens_cx", "lens_cy")
_ENV_MODE = {"vacuum": "真空 (n=1)", "air": "大気 (Air.nk)", "custom": "カスタム数値"}


def _complex(v):
    if isinstance(v, dict):
        return complex(v.get("re", 0.0) or 0.0, v.get("im", 0.0) or 0.0)
    return complex(v)


def _renumber(items, counter_key, ss):
    """items を深いコピーし、id を新しく振る（カウンタ ss[counter_key] を進める）。"""
    out = copy.deepcopy(list(items or []))
    n = int(ss.get(counter_key, 0) or 0)
    for it in out:
        n += 1
        it["id"] = n
    ss[counter_key] = n
    return out


def _materials_in(obj):
    """層・レシピの入れ子から材質名（material / bg_material / mat_a / mat_b）を集める。"""
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("material", "bg_material", "mat_a", "mat_b") and isinstance(v, str):
                found.add(v)
            else:
                found |= _materials_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _materials_in(v)
    return found


def _check_materials(names, nk_namelist, warn):
    missing = sorted(n for n in names if n not in nk_namelist and n not in _SENTINELS)
    if missing:
        warn.append(f"この PC に無い材質があります: {', '.join(missing)}（data/nk に .nk を置いてから読み込み直してください）")


def restore_2d(ss, params, nk_namelist):
    """2D モードの入力欄へ書き戻す。"""
    warn = []
    sb = params.get("sidebar") or {}
    shape = sb.get("lattice_shape", "正方格子")
    ss["lattice_shape"] = shape
    ss["pitch_x"] = float(sb.get("pitch_x_nm", 400.0))
    if shape == "矩形格子":
        ss["pitch_y"] = float(sb.get("pitch_y_nm", 400.0))

    env = sb.get("env_spec", 1.0)
    if env == "Air":
        ss["env_mode"] = _ENV_MODE["air"]
    elif isinstance(env, (int, float)) and float(env) == 1.0:
        ss["env_mode"] = _ENV_MODE["vacuum"]
    else:
        c = _complex(env)
        ss.update(env_mode=_ENV_MODE["custom"], env_n_custom=float(c.real), env_k_custom=float(c.imag))

    subst = sb.get("subst_spec", "SiO2")
    if isinstance(subst, str):
        if subst in nk_namelist:
            ss.update(subst_mode="材質一覧から選択", subst_material=subst)
        else:
            warn.append(f"基板の材質 {subst} がこの PC にありません（基板はいまの設定のまま）。")
    else:
        c = _complex(subst)
        ss.update(subst_mode="カスタム数値", a2d_subst_n_custom=float(c.real), a2d_subst_k_custom=float(c.imag))

    mode = sb.get("angle_mode", "単一角度")
    ss["a2d_angle_mode"] = mode
    ss["phi_deg"] = float(sb.get("phi_deg", 0.0))
    ss["pol_deg"] = float(sb.get("pol_deg", 0.0))
    if mode == "単一角度":
        ss["theta_deg"] = float(sb.get("theta_deg", 0.0))
    else:
        lo, hi = (float(v) for v in sb.get("theta_range", (0.0, 60.0)))
        n = int(sb.get("n_theta", 1))
        ss["theta_range"] = (lo, hi)
        # 計算は linspace(lo, hi, n_theta) なので、点数が同じになる刻みなら同じ角度になる
        ss["theta_step"] = float(sb.get("theta_step") or ((hi - lo) / (n - 1) if n > 1 else 10.0))
    ss["wl_min"] = float(sb.get("wl_min_nm", 380.0))
    ss["wl_max"] = float(sb.get("wl_max_nm", 780.0))
    ss["n_freq"] = int(sb.get("n_freq", 21))
    ss["norder"] = int(sb.get("norder", 49))

    ss["array_layers"] = _renumber(params.get("array_layers"), "_array_layer_id", ss)
    step, lens = params.get("step_regions"), params.get("lens")
    ss["lens_enabled"] = lens is not None
    ss["step_enabled"] = step is not None and lens is None
    if step is not None and lens is None:
        ss["step_regions"] = _renumber(step, "_step_region_id", ss)
        ss["step_film_layers"] = _renumber(params.get("step_film"), "_step_layer_id", ss)
    if lens is not None:
        spec, recipe = lens
        for k in _LENS_KEYS:
            ss.pop(k, None)
        ss["lens_spec"] = dict(spec)
        ss["lens_film_layers"] = _renumber(recipe, "_lens_layer_id", ss)
    _restore_supercell(ss, params.get("supercell"), nk_namelist, warn)
    _check_materials(_materials_in([params.get("array_layers"), params.get("step_film"), lens]), nk_namelist, warn)
    return warn


def _restore_supercell(ss, sc, nk_namelist, warn):
    """周期領域と円柱の切り抜き（iridis_app/array2d_supercell_editor.py の入力欄）を戻す。"""
    ss["sc_enabled"] = bool(sc)
    if not sc:
        return
    nx, ny = int(sc.get("nx", 1)), int(sc.get("ny", 1))
    preset = {(1, 1): "1×1", (3, 3): "3×3"}.get((nx, ny), "任意（Nx×Ny）")
    ss["sc_preset"] = preset
    if preset.startswith("任意"):
        ss["sc_nx"], ss["sc_ny"] = nx, ny
    ss["sc_cut"] = bool(sc.get("cut"))
    if not sc.get("cut"):
        return
    ss["sc_mode"] = sc.get("mode", "keep")
    out = sc.get("outside_material") or ENV_MATERIAL
    label = {ENV_MATERIAL: "環境と同じ", SUBST_MATERIAL: "基板と同じ"}.get(out, out)   # material_selectbox の選択肢
    if label in ("環境と同じ", "基板と同じ") or out in nk_namelist:
        ss["sc_out"] = label
    else:
        warn.append(f"切り抜きの外側の材質 {out} がこの PC にありません（外側の材質はいまの設定のまま）。")
    ss["sc_gap"] = float(sc.get("gap_nm", 0.0))
    ss["sc_r_auto"] = bool(sc.get("r_auto", False))
    if not ss["sc_r_auto"]:
        ss["sc_r"] = float(sc.get("radius_nm", 100.0))
    ss["sc_cx"], ss["sc_cy"] = float(sc.get("cx_nm", 0.0)), float(sc.get("cy_nm", 0.0))
    ss["sc_depth"] = float(sc.get("subst_depth_nm", 0.0))


def restore_1d(ss, params, nk_namelist):
    """1D モードの入力欄へ書き戻す。"""
    warn = []
    ss["angle_mode"] = params.get("angle_mode", "Single angle (2D)")
    if ss["angle_mode"] == "Single angle (2D)":
        ss["inc_angle"] = float(params.get("inc_angle", 0.0))
    else:
        ss["angle_range"] = tuple(float(v) for v in params.get("angle_range", (0.0, 80.0)))
        ss["angle_step"] = float(params.get("angle_step", 2.0))
    ss["wl_option"] = params.get("wl_option", "Visible[380-780nm]")
    if ss["wl_option"] == "Any":
        ss["wl_range"] = (float(params["wl_min"]), float(params["wl_max"]))
        ss["wln"] = int(params["wl_n"])
    ss["nenv"] = float(params.get("n_env", 1.0))
    sub = params.get("substrate")
    if sub in nk_namelist:
        ss["substrate"] = sub
    elif sub:
        warn.append(f"基板の材質 {sub} がこの PC にありません（基板はいまの設定のまま）。")
    custom = bool(params.get("custom_subst_nk"))
    ss["custom_subst_nk"] = custom
    if custom:
        ss["subst_n_custom"] = float(params.get("subst_n_custom") or 3.0)
        ss["subst_k_custom"] = float(params.get("subst_k_custom") or 0.0)
    ss["nOrder"] = int(params.get("norder", 11))
    if params.get("graded_n_slices") is not None:
        ss["graded_n_slices"] = int(params["graded_n_slices"])
    ro = int(params.get("repr_order", 0))
    if abs(ro) <= ss["nOrder"] // 2:
        ss["repr_order"] = ro

    cols = copy.deepcopy(list(params.get("columns") or []))
    n_col, n_lay = int(ss.get("_ncol_id", 0) or 0), int(ss.get("_nlayer_id", 0) or 0)
    for col in cols:
        n_col += 1
        col["id"] = n_col
        for fl in col.get("film_layers", []):
            n_lay += 1
            fl["id"] = n_lay
    ss.update(columns=cols, _ncol_id=n_col, _nlayer_id=n_lay)
    _check_materials(_materials_in(cols), nk_namelist, warn)
    return warn


_GRID_KEYS = ("wl_nm", "theta_deg", "angle_ar")


def compare_results(saved, new):
    """保存した結果と再計算した結果（どちらも to_jsonable 済みの dict）を比べる。
    戻り値: {"grid_ok": 波長・角度が同じか, "diffs": {量: 最大の差}, "max": 全体の最大の差}。
    比べるのは、両方にあって形が同じ数値配列だけ（NaN 同士は差 0、片方だけ NaN は差 inf）。"""
    grid_ok = True
    for k in _GRID_KEYS:
        if k in saved and k in new:
            a, b = from_jsonable_array(saved[k]), from_jsonable_array(new[k])
            grid_ok = grid_ok and a.shape == b.shape and bool(np.allclose(a, b))
    diffs = {}
    for k, v in saved.items():
        if k in _GRID_KEYS or k not in new or not isinstance(v, list):
            continue
        try:
            a, b = from_jsonable_array(v).astype(float), from_jsonable_array(new[k]).astype(float)
        except (TypeError, ValueError):
            continue
        if a.shape != b.shape or a.size == 0:
            continue
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        diffs[k] = float(np.max(np.where(np.isnan(d), np.inf, d)))
    return {"grid_ok": grid_ok, "diffs": diffs, "max": max(diffs.values()) if diffs else None}


def nk_differences(saved_snapshot):
    """保存時の nk ファイルの中身と、いまの data/nk の中身の違い（材質名のリスト: 変わった, 無い）。"""
    now = nk_snapshot(list(saved_snapshot or {}))
    changed = sorted(n for n, text in (saved_snapshot or {}).items() if n in now and now[n].strip() != text.strip())
    missing = sorted(n for n in (saved_snapshot or {}) if n not in now)
    return changed, missing

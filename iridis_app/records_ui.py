"""解析記録（Supabase）の画面部品。

- render_auth_sidebar(): サイドバーのログイン欄（全モード共通）
- render_restore_notice(mode): 記録から条件を読み込んだ直後の案内（1D/2D 画面の上）
- consume_autorun(mode): 「読み込んで再計算」のとき、計算ボタンの代わりに True を 1 回返す
- render_save_panel(mode, payload): 計算結果の下に出す「💾 解析記録に保存」（再計算なら保存時の結果との比較も）
- render_records_page(): 「📚 解析記録」モードの画面（一覧・絞り込み・詳細・比較・編集・削除・テーマ・
  図と添付ファイル・条件の読み込み）

接続先は .streamlit/secrets.toml の [supabase] url / anon_key（または環境変数 IRIDIS_SUPABASE_URL /
IRIDIS_SUPABASE_ANON_KEY）。RecordStore はブラウザセッションごとに session_state に持つ
（ログイン状態を他の利用者と共有しないため。st.cache_resource は使わない）。
保存する中身の形（payload）:
  {"params": 計算条件, "result": 結果, "summary": {"text": 一覧用の一行, ...}, "materials": 材質名,
   "elapsed_s": 秒, "calc_id": 計算ごとに一意な文字列, "calc_time": 計算を始めた time.time(),
   "default_title": 題名の初期値, "figures_fn": 構造の図 {名前: plotly Figure} を返す関数（任意）}
"""
import datetime
import json
import os
import time

import pandas as pd
import plotly.io as pio
import streamlit as st

from RCWA2D.color_display import render_gsolver_style_display

from .common import TAB_KEYS, TAB_STRUCT, get_nk_list, show_result_tab
from .log import get_logger
from .records import (RecordStore, RecordsError, cached_git_info, from_jsonable_array, guess_type, make_record,
                      start_git_info, to_jsonable)
from .records_figures import (QUANTITIES, angles, figure_files, heatmap_fig, result_figures, result_pngs,
                              spectra_fig, spectrum)
from .records_restore import compare_results, nk_differences, restore_1d, restore_2d

_MODE_LABEL = {"1d": "1D 列ベース", "2d": "2D 周期アレイ"}
APP_MODE = {"1d": "列ベース構造 (1D)", "2d": "2次元周期アレイ (2D)"}   # iridis.py の構造タイプの選択肢
JST = datetime.timezone(datetime.timedelta(hours=9))
_log = get_logger("iridis.records")
SAME_TOL = 1e-6   # 再計算が保存時の結果と「一致」とみなす差（結果は有効数字 9 桁で保存している）


# ---------------------------------------------------------------------------
# 接続・ログイン
# ---------------------------------------------------------------------------
def make_store(url, anon_key):
    """テストではこれを差し替える。"""
    return RecordStore.connect(url, anon_key)


def _config():
    url = os.environ.get("IRIDIS_SUPABASE_URL")
    key = os.environ.get("IRIDIS_SUPABASE_ANON_KEY")
    if url and key:
        return url, key
    try:
        sec = st.secrets.get("supabase", {})
        return sec.get("url"), sec.get("anon_key")
    except Exception:   # secrets.toml が無い
        return None, None


def get_store():
    """このブラウザセッションの RecordStore。未設定なら None。"""
    if "records_store" not in st.session_state:
        url, key = _config()
        if not (url and key):
            return None
        try:
            st.session_state.records_store = make_store(url, key)
        except Exception as e:
            st.session_state.records_store = None
            st.session_state.records_connect_error = str(e)
    return st.session_state.records_store


def current_user():
    """ログイン中なら {"id", "email", "member"}。"""
    store = get_store()
    if store is None:
        return None
    user = store.current_user()
    if user is None:
        return None
    member = st.session_state.get("records_member")
    if member is None or member[0] != user["id"]:
        try:
            member = (user["id"], store.is_member())
        except RecordsError:
            member = (user["id"], False)
        st.session_state.records_member = member
    return dict(user, member=member[1])


# ボタンの処理は on_click のコールバックで行う（画面を描き直す前に実行されるので st.rerun が要らない）。
# コールバック内のメッセージは _flash で預け、描画中に _show_flash で出す。
def _flash(where, kind, text):
    st.session_state[f"records_flash_{where}"] = (kind, text)


def _show_flash(where):
    msg = st.session_state.pop(f"records_flash_{where}", None)
    if msg:
        getattr(st, msg[0])(msg[1])


def _do_login(store):
    try:
        store.sign_in(st.session_state.get("records_email", ""), st.session_state.get("records_pw", ""))
        st.session_state.pop("records_member", None)
        _log.info("解析記録: ログイン")
    except RecordsError as e:
        _log.warning("解析記録: ログインできなかった（%s）", e)
        _flash("login", "error", str(e))


def _do_logout(store):
    store.sign_out()
    for k in ("records_member", "records_saved", "records_file_cache"):
        st.session_state.pop(k, None)


def render_auth_sidebar():
    start_git_info()   # 保存時に使うコードの版を、起動直後から裏で取っておく
    with st.sidebar.expander("🔐 解析記録（ログイン）", expanded=False):
        store = get_store()
        if store is None:
            err = st.session_state.get("records_connect_error")
            st.caption(f"接続できません: {err}" if err else
                       "未設定です。supabase/README.md の手順で .streamlit/secrets.toml に接続先を書くと使えます。")
            return
        user = current_user()
        if user is None:
            with st.form("records_login", border=False):
                st.text_input("メールアドレス", key="records_email")
                st.text_input("パスワード", type="password", key="records_pw")
                st.form_submit_button("ログイン", width="stretch", on_click=_do_login, args=(store,))
            _show_flash("login")
            st.caption("アカウントは管理者が作ります（supabase/README.md）。ページを再読み込みするとログアウトします。")
            return
        st.markdown(f"ログイン中: **{user['email']}**")
        if not user["member"]:
            st.warning("研究室メンバーに登録されていないため、記録の閲覧・保存はできません。管理者に連絡してください。")
        st.button("ログアウト", width="stretch", key="records_logout", on_click=_do_logout, args=(store,))
        with st.popover("パスワードを変更", width="stretch"):
            p1 = st.text_input("新しいパスワード（8 文字以上）", type="password", key="records_pw_new")
            p2 = st.text_input("もう一度", type="password", key="records_pw_new2")
            if st.button("変更する", key="records_pw_change"):
                if p1 != p2:
                    st.error("2 つの入力が違います。")
                else:
                    try:
                        store.change_password(p1)
                        st.success("変更しました。")
                    except RecordsError as e:
                        st.error(str(e))


# ---------------------------------------------------------------------------
# 条件の読み込み（記録 → 画面）
# ---------------------------------------------------------------------------
def _do_restore(row, autorun):
    """記録の条件を 1D/2D 画面の入力欄へ書き戻し、その画面へ切り替える（on_click で呼ぶ）。"""
    mode = row["mode"]
    ss = st.session_state
    try:
        nk = get_nk_list()
        warn = (restore_2d if mode == "2d" else restore_1d)(ss, row["params"], nk)
    except (KeyError, TypeError, ValueError) as e:
        _flash("detail", "error", f"この記録の条件は読み込めません（{type(e).__name__}: {e}）。")
        return
    changed, missing = nk_differences(row.get("nk_snapshot"))
    if changed:
        warn.append(f"nk ファイルの中身が保存したときと違います: {', '.join(changed)}（再計算の結果がずれることがあります）")
    if missing:
        warn.append(f"保存したときの nk ファイルがこの PC にありません: {', '.join(missing)}")
    commit, _ = cached_git_info(wait_s=0)
    if row.get("git_commit") and commit and commit != row["git_commit"]:
        warn.append(f"コードの版が違います（保存時 {row['git_commit'][:8]}、いま {commit[:8]}）。")
    if row.get("git_dirty"):
        warn.append("保存したときのコードには未コミットの変更がありました（完全には再現できないことがあります）。")
    ss["app_mode"] = APP_MODE[mode]
    ss["records_restored"] = {"id": row["id"], "title": row["title"], "mode": mode, "result": row["result"],
                              "t": time.time()}
    if autorun:
        ss["records_autorun"] = mode
        show_result_tab(mode)
    else:
        ss[TAB_KEYS[mode]] = TAB_STRUCT
    ss["records_flash_restore"] = (mode, row["title"], warn, autorun)


def render_restore_notice(mode):
    msg = st.session_state.get("records_flash_restore")
    if not msg or msg[0] != mode:
        return
    st.session_state.pop("records_flash_restore")
    _, title, warn, autorun = msg
    st.info(f"📚 記録「{title}」の条件を読み込みました。" +
            ("計算を始めました。終わると下に保存時の結果との比較が出ます。" if autorun else
             "「▶ 計算」を押すと再計算できます（結果の下に保存時の結果との比較が出ます）。"))
    for w in warn:
        st.warning(w)


def consume_autorun(mode):
    """「読み込んで再計算」の直後だけ True（1 回で消える）。"""
    if st.session_state.get("records_autorun") == mode:
        del st.session_state["records_autorun"]
        return True
    return False


def _render_reproduce_check(mode, payload):
    """記録から読み込んだ条件で計算したあとなら、保存時の結果との差を出す。"""
    rest = st.session_state.get("records_restored")
    if not rest or rest["mode"] != mode or payload.get("calc_time", 0) < rest["t"]:
        return
    cmp = compare_results(rest["result"], to_jsonable(payload["result"], digits=9))
    if not cmp["grid_ok"] or cmp["max"] is None:
        st.warning(f"📚 記録「{rest['title']}」と波長・角度の点が違うため、結果を比べられません。")
    elif cmp["max"] <= SAME_TOL:
        st.success(f"📚 記録「{rest['title']}」と同じ結果になりました（最大の差 {cmp['max']:.1e}）。")
    else:
        st.warning(f"📚 記録「{rest['title']}」と結果が違います（最大の差 {cmp['max']:.3g}）。"
                   "nk ファイル・コードの版・条件の変更を確かめてください。")
        st.dataframe(pd.DataFrame({"最大の差": cmp["diffs"]}), width="content")


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def _projects(store):
    try:
        return store.list_projects()
    except RecordsError as e:
        st.error(str(e))
        return []


def _figure_uploads(rec, payload, title):
    """保存する図: 結果の PNG（matplotlib）と HTML/JSON、構造の HTML/JSON。{ファイル名: (bytes, 種類)}。"""
    row = {"mode": rec["mode"], "title": title, "result": rec["result"]}
    files = {name: (data, "image/png") for name, data in result_pngs(row).items()}
    figs = dict(result_figures(row))
    if payload.get("figures_fn"):
        figs.update(payload["figures_fn"]())
    for name, fig in figs.items():
        files.update(figure_files(name, fig))
    return files


def _upload_all(store, run_id, figures, attachments):
    """図と添付ファイルを付ける。1 つ失敗しても残りは続け、失敗したものを返す。"""
    errors, n = [], 0
    for name, (data, ctype) in figures.items():
        try:
            store.upload_file(run_id, name, data, ctype, kind="figure")
            n += 1
        except RecordsError as e:
            errors.append(str(e))
    for f in attachments or []:
        try:
            store.upload_file(run_id, f.name, f.getvalue(), guess_type(f.name, f.type), kind="attachment")
            n += 1
        except RecordsError as e:
            errors.append(str(e))
    return n, errors


def render_save_panel(mode, payload):
    """計算結果の下に置く保存欄。payload の形はモジュールの説明を参照。"""
    store = get_store()
    _render_reproduce_check(mode, payload)
    if store is None:
        return
    saved = st.session_state.setdefault("records_saved", {})
    with st.expander("💾 この結果を解析記録に保存", expanded=False):
        user = current_user()
        if user is None:
            st.info("サイドバーの「🔐 解析記録（ログイン）」からログインすると保存できます。")
            return
        if not user["member"]:
            st.warning("研究室メンバーに登録されていないため保存できません。")
            return
        if payload.get("calc_id") in saved:
            st.success(f"この計算結果は保存済みです（記録 ID: {saved[payload['calc_id']]}）。"
                       "「📚 解析記録」で確認できます。")
            return
        if "_error" in (payload.get("params") or {}):
            st.error(f"計算条件を保存用に変換できませんでした: {payload['params']['_error']}")
            return
        st.caption(f"条件: {payload.get('summary', {}).get('text', '')}")
        cid = payload.get("calc_id", "x")
        title = st.text_input("題名", payload.get("default_title", ""), max_chars=200, key=f"rec_title_{cid}")
        memo = st.text_area("メモ（目的・気づいたことなど）", key=f"rec_memo_{cid}", height=80)
        c1, c2 = st.columns(2)
        with c1:
            tags = st.text_input("タグ（カンマ区切り）", key=f"rec_tags_{cid}", placeholder="例: TiO2, 円盤, 収束確認済み")
        with c2:
            projects = _projects(store)
            names = ["（テーマなし）"] + [p["name"] for p in projects]
            choice = st.selectbox("テーマ", names, key=f"rec_project_{cid}",
                                  help="テーマは「📚 解析記録」画面で作れます。")
        with_figs = st.checkbox("図も保存する（結果の PNG と、拡大・回転できる HTML。構造の図も）", value=True,
                                key=f"rec_figs_{cid}")
        attachments = st.file_uploader("添付ファイル（SEM 像・実測スペクトルなど。画像・PDF・CSV・テキスト、1 つ 20 MB まで）",
                                       accept_multiple_files=True, key=f"rec_files_{cid}")
        if st.button("💾 保存", type="primary", key=f"rec_save_{cid}"):
            project_id = next((p["id"] for p in projects if p["name"] == choice), None)
            try:
                rec = make_record(mode, title, payload["params"], payload["result"], payload.get("summary"),
                                  payload.get("materials", ()), memo=memo, tags=tags, project_id=project_id,
                                  elapsed_s=payload.get("elapsed_s"))
                run_id = store.save_run(rec)
            except (RecordsError, TypeError, ValueError) as e:
                _log.warning("解析記録: 保存できなかった（%s）", e)
                st.error(str(e))
                return
            _log.info("解析記録: 保存 %s（%s）", run_id, mode)
            saved[cid] = run_id
            figures, fig_error = {}, None
            if with_figs:
                try:
                    figures = _figure_uploads(rec, payload, title)
                except Exception as e:   # 図が作れなくても記録そのものは保存済み
                    fig_error = f"図を作れませんでした: {type(e).__name__}: {e}"
            with st.spinner("図・添付ファイルをアップロード中..."):
                n, errors = _upload_all(store, run_id, figures, attachments)
            st.success(f"保存しました（記録 ID: {run_id}、図・添付ファイル {n} 件）。「📚 解析記録」で確認できます。")
            for err in ([fig_error] if fig_error else []) + errors:
                st.warning(err)


# ---------------------------------------------------------------------------
# 比較用
# ---------------------------------------------------------------------------
def _flatten(x, prefix=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(x, list) and x and all(isinstance(v, dict) for v in x):
        out = {}
        for i, v in enumerate(x):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x}


def params_diff(rows):
    """複数の記録の計算条件のうち、値が違う項目だけの表（行=項目、列=記録）。"""
    flats = [_flatten(r["params"]) for r in rows]
    keys = sorted(set().union(*flats))
    diff = {k: [f.get(k, "—") for f in flats] for k in keys
            if len({json.dumps(f.get(k, "—"), ensure_ascii=False, default=str) for f in flats}) > 1}
    return pd.DataFrame(diff, index=[r["title"] for r in rows]).T


def _local_time(ts):
    try:
        return datetime.datetime.fromisoformat(ts).astimezone(JST).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(ts)


# ---------------------------------------------------------------------------
# 「📚 解析記録」画面
# ---------------------------------------------------------------------------
def render_records_page():
    st.title("Iridis — 📚 解析記録")
    _show_flash("page")
    store = get_store()
    if store is None:
        st.info("解析記録のデータベースが未設定です。supabase/README.md の手順で設定してください。")
        return
    user = current_user()
    if user is None:
        st.info("サイドバーの「🔐 解析記録（ログイン）」からログインしてください。")
        return
    if not user["member"]:
        st.warning("研究室メンバーに登録されていないため、記録を見られません。管理者に連絡してください。")
        return

    projects = _projects(store)
    pname = {p["id"]: p["name"] for p in projects}
    _render_projects(store, projects)

    f1, f2, f3, f4, f5 = st.columns([1, 1.4, 1.2, 1.4, 1])
    with f1:
        mode = st.selectbox("モード", ["すべて", "1D", "2D"], key="rec_f_mode")
    with f2:
        proj = st.selectbox("テーマ", ["すべて"] + [p["name"] for p in projects], key="rec_f_project")
    with f3:
        tag = st.text_input("タグ", key="rec_f_tag")
    with f4:
        text = st.text_input("題名に含む語", key="rec_f_text")
    with f5:
        mine = st.checkbox("自分の記録だけ", key="rec_f_mine")
    try:
        rows = store.list_runs(mode=None if mode == "すべて" else mode.lower(),
                               project_id=next((p["id"] for p in projects if p["name"] == proj), None),
                               tag=tag.strip() or None, text=text.strip() or None,
                               user_id=user["id"] if mine else None)
    except RecordsError as e:
        st.error(str(e))
        return
    if not rows:
        st.info("条件に合う記録はありません。計算結果の下の「💾 この結果を解析記録に保存」から保存できます。")
        return

    st.dataframe(pd.DataFrame([{
        "日時": _local_time(r["created_at"]), "題名": r["title"], "モード": r["mode"].upper(),
        "作成者": r["owner_email"], "タグ": ", ".join(r["tags"]), "テーマ": pname.get(r["project_id"], ""),
        "条件": (r.get("summary") or {}).get("text", ""), "メモ": r["memo"],
    } for r in rows]), hide_index=True, width="stretch")
    st.caption(f"{len(rows)} 件（新しい順、最大 200 件）")

    label = {r["id"]: f"{_local_time(r['created_at'])}  {r['title']}  ({r['owner_email']})" for r in rows}
    # 削除した記録や絞り込みで一覧から外れた記録は選択からも外す（選択欄を作る前でないと書き換えられない）
    if "rec_pick" in st.session_state:
        valid = [i for i in st.session_state.rec_pick if i in label]
        if valid != st.session_state.rec_pick:
            st.session_state.rec_pick = valid
    picked = st.multiselect("開く記録（2 件以上選ぶと重ねて比較）", [r["id"] for r in rows],
                            format_func=label.get, key="rec_pick")
    if not picked:
        return
    try:
        full = [store.get_run(i) for i in picked]
    except RecordsError as e:
        st.error(str(e))
        return
    if len(full) == 1:
        _render_detail(store, full[0], user, projects)
    else:
        _render_compare(full)


def _render_projects(store, projects):
    with st.expander("🗂 テーマ（記録のまとまり）", expanded=False):
        if projects:
            st.dataframe(pd.DataFrame([{"テーマ": p["name"], "説明": p["description"]} for p in projects]),
                         hide_index=True, width="stretch")
        c1, c2, c3 = st.columns([1.2, 2, 0.8])
        with c1:
            st.text_input("新しいテーマ名", key="rec_new_project")
        with c2:
            st.text_input("説明", key="rec_new_project_desc")
        with c3:
            st.write("")
            st.button("作成", key="rec_create_project", on_click=_do_create_project, args=(store,))
        _show_flash("project")


def _do_create_project(store):
    try:
        store.create_project(st.session_state.get("rec_new_project", ""), st.session_state.get("rec_new_project_desc", ""))
        st.session_state.rec_new_project = st.session_state.rec_new_project_desc = ""
    except RecordsError as e:
        _flash("project", "error", str(e))


def _do_update(store, rid, projects):
    ss = st.session_state
    choice = ss.get(f"rec_e_proj_{rid}")
    try:
        store.update_run(rid, title=ss.get(f"rec_e_title_{rid}", ""), memo=ss.get(f"rec_e_memo_{rid}", ""),
                         tags=ss.get(f"rec_e_tags_{rid}", ""),
                         project_id=next((p["id"] for p in projects if p["name"] == choice), None))
        _flash("detail", "success", "変更しました。")
    except RecordsError as e:
        _flash("detail", "error", str(e))


def _do_delete(store, rid):
    try:
        store.delete_run(rid)
        _log.info("解析記録: 削除 %s", rid)
        _flash("page", "success", "削除しました（図・添付ファイルも消しました）。")
    except RecordsError as e:
        _flash("detail", "error", str(e))


def _do_attach(store, rid, key):
    ss = st.session_state
    n, errors = _upload_all(store, rid, {}, ss.get(key))
    ss[f"rec_up_n_{rid}"] = ss.get(f"rec_up_n_{rid}", 0) + 1   # 選択欄を空に戻す（key を変える）
    _flash("files", "warning" if errors else "success",
           " / ".join(errors) if errors else f"{n} 件を添付しました。")


def _do_delete_file(store, frow):
    try:
        store.delete_file(frow)
        st.session_state.get("records_file_cache", {}).pop(frow["path"], None)
    except RecordsError as e:
        _flash("files", "error", str(e))


def _file_bytes(store, path):
    cache = st.session_state.setdefault("records_file_cache", {})
    if path not in cache:
        cache[path] = store.download_file(path)
    return cache[path]


def _render_files(store, row, own):
    rid = row["id"]
    try:
        files = store.list_files(rid)
    except RecordsError as e:
        st.error(str(e))
        return
    st.markdown(f"**図・添付ファイル**（{len(files)} 件）")
    _show_flash("files")
    if files:
        st.dataframe(pd.DataFrame([{"ファイル": f["name"], "種類": "図" if f["kind"] == "figure" else "添付",
                                    "大きさ": f"{f['size_bytes'] / 1024:.0f} KB"} for f in files]),
                     hide_index=True, width="stretch")
        if st.toggle("図・添付ファイルを表示する（ダウンロードして表示）", key=f"rec_showfiles_{rid}"):
            for f in files:
                try:
                    data = _file_bytes(store, f["path"])
                except RecordsError as e:
                    st.error(str(e))
                    continue
                c1, c2 = st.columns([5, 1])
                with c1:
                    if f["content_type"].startswith("image/"):
                        st.image(data, caption=f["name"])
                    elif f["kind"] == "figure" and f["name"].endswith(".json"):
                        st.plotly_chart(pio.from_json(data.decode("utf-8")), width="stretch",
                                        key=f"rec_fig_{f['id']}")
                    else:
                        st.caption(f["name"])
                with c2:
                    st.download_button("⬇", data, f["name"], f["content_type"], key=f"rec_fdl_{f['id']}",
                                       help=f"{f['name']} をダウンロード")
                    if own:
                        st.button("✕", key=f"rec_frm_{f['id']}", help=f"{f['name']} を削除",
                                  on_click=_do_delete_file, args=(store, f))
    if own:
        key = f"rec_up_{rid}_{st.session_state.get(f'rec_up_n_{rid}', 0)}"
        st.file_uploader("ファイルを添付する（画像・PDF・CSV・テキスト、1 つ 20 MB まで）", accept_multiple_files=True,
                         key=key)
        st.button("📎 添付する", key=f"rec_attach_{rid}", on_click=_do_attach, args=(store, rid, key),
                  disabled=not st.session_state.get(key))


# 1D 画面と同じ「表示する光成分」の選択肢
def _order_label(m):
    return "0 (specular)" if m == 0 else f"{m:+d}"


def _render_result_like_calc(row):
    """保存した結果を、計算画面と同じ形で表示する。
    角度掃引: Gsolver'S COLOR.html 互換表示（スウォッチ 0/30/60°＋角度チャート＋スペクトルチャート）。
      途中で止めた計算（全点がそろっていない）だけは、計算画面と同じく簡易ヒートマップ。
    単一角度: 計算画面と同じ折れ線（2D は R0・R_total・T0・T_total、1D は指定次数の Rp・Rs）。"""
    rid, r = row["id"], row["result"]
    th = angles(row)
    wl = from_jsonable_array(r["wl_nm"])
    cond = (row.get("summary") or {}).get("text", "")
    if row["mode"] == "2d":
        if len(th) == 1:
            st.plotly_chart(spectra_fig([(name, *spectrum(row, q)[:2]) for name, q in QUANTITIES["2d"].items()],
                                        f"θ={th[0]:g}°, {cond}"), width="stretch")
            return
        qname = st.selectbox("表示する量", list(QUANTITIES["2d"]), key=f"rec_q_{rid}")
        q = QUANTITIES["2d"][qname]
        z, sw = from_jsonable_array(r[q]), r.get("swatch")
        complete = sw is not None and not (row.get("summary") or {}).get("partial")
        sw_th, sw_vals = (sw["theta_deg"], from_jsonable_array(sw[q])) if sw is not None else ([], [])
        show_sw = True
    else:
        label = _order_label(r.get("repr_order", 0))
        if r["mode"] == "single":
            series = [(f"{p} (m={label})", *spectrum(row, p)[:2]) for p in ("Rp", "Rs")]
            st.plotly_chart(spectra_fig(series, f"AOI {th[0]:g} deg", mode="lines"), width="stretch")
            return
        c1, c2 = st.columns([1, 3])
        with c1:
            pol = st.radio("偏光 / Polarization", ("Rp", "Rs"), horizontal=True, key=f"rec_pol_{rid}")
        modes = {f"指定次数 (m={label})": "single", "0次光以外 (回折光 m≠0 合算)": "diff",
                 "全次数合算 (0次光含む)": "total"}
        with c2:
            qname = st.radio("表示する光成分", list(modes), index=2, horizontal=True, key=f"rec_ord_{rid}")
        suffix = modes[qname]
        key = {"single": f"{pol}2d", "diff": f"{pol}_diff_2d", "total": f"{pol}_total_2d"}[suffix]
        z = from_jsonable_array(r[key])
        show_sw = bool(r.get("show_swatches"))
        sw_th = r.get("swatch_thetas", [])
        sw_vals = from_jsonable_array(r["swatch"][f"{pol}_{suffix}"]) if show_sw else []
        complete = True
        qname = f"{pol} {qname}"
    if complete:
        n_filled = render_gsolver_style_display(lambdas=wl, thetas=th, values=z, show_swatches=show_sw,
                                                swatch_thetas=list(sw_th), swatch_values=sw_vals)
        if n_filled:
            st.warning(f"S4 が数値的に解けなかった点が {n_filled} 点あり、表示では同じ角度の隣の波長から"
                       "線形補間しています。")
        st.caption(f"{qname} — {cond}")
    else:
        st.info("途中で打ち切られた計算なので、Gsolver'S COLOR.html 互換表示ではなく簡易ヒートマップで表示しています。")
        st.plotly_chart(heatmap_fig(row, qname, q), width="stretch")


def _render_detail(store, row, user, projects):
    st.subheader(row["title"])
    s = row.get("summary") or {}
    st.caption(f"{_MODE_LABEL[row['mode']]} ・ {_local_time(row['created_at'])} ・ {row['owner_email']} ・ "
               f"コード {str(row.get('git_commit') or '不明')[:8]}{'（未コミットの変更あり）' if row.get('git_dirty') else ''}"
               + (f" ・ 計算 {row['elapsed_s']:.0f} 秒" if row.get("elapsed_s") else ""))
    if s.get("text"):
        st.markdown(f"**条件**: {s['text']}")
    if row["memo"]:
        st.markdown(f"**メモ**: {row['memo']}")
    if s.get("partial"):
        st.warning("この記録は途中で停止した計算の部分的な結果です。")
    _show_flash("detail")

    r1, r2 = st.columns(2)
    with r1:
        st.button("↩ 条件を画面に読み込む", key=f"rec_restore_{row['id']}", width="stretch",
                  on_click=_do_restore, args=(row, False),
                  help=f"{_MODE_LABEL[row['mode']]} の画面に、この記録の計算条件（構造・入射条件・NumBasis など）を入れる。")
    with r2:
        st.button("↩ 読み込んで再計算", key=f"rec_rerun_{row['id']}", type="primary", width="stretch",
                  on_click=_do_restore, args=(row, True),
                  help="条件を読み込んですぐ計算し、保存時の結果と同じになるか確かめる。")

    th = angles(row)
    qmap = QUANTITIES[row["mode"]]
    _render_result_like_calc(row)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("記録をまるごとダウンロード（JSON）", json.dumps(row, ensure_ascii=False, indent=1),
                           f"iridis_record_{row['id'][:8]}.json", "application/json", key=f"rec_dl_{row['id']}")
    with d2:
        cols = {"Wavelength(nm)": from_jsonable_array(row["result"]["wl_nm"])}
        for t in th:
            for name, q in qmap.items():
                cols[f"{q} θ={t:g}"] = spectrum(row, q, t)[1]
        st.download_button("スペクトル CSV（全角度・全量）", pd.DataFrame(cols).to_csv(index=False).encode("utf-8"),
                           f"iridis_record_{row['id'][:8]}.csv", "text/csv", key=f"rec_csv_{row['id']}")

    own = row["user_id"] == user["id"]
    _render_files(store, row, own)
    with st.expander("計算条件（params）", expanded=False):
        st.json(row["params"], expanded=2)
    with st.expander(f"使った nk ファイルの中身（{len(row.get('nk_snapshot') or {})} 件）", expanded=False):
        for name, text in (row.get("nk_snapshot") or {}).items():
            st.markdown(f"**{name}.nk**")
            st.code(text, language=None)

    if not own:
        st.caption("他のメンバーの記録なので、題名・メモの変更や削除はできません。")
        return
    with st.expander("✏️ 題名・メモ・タグ・テーマを変更／削除", expanded=False):
        rid = row["id"]
        st.text_input("題名", row["title"], key=f"rec_e_title_{rid}")
        st.text_area("メモ", row["memo"], key=f"rec_e_memo_{rid}")
        st.text_input("タグ（カンマ区切り）", ", ".join(row["tags"]), key=f"rec_e_tags_{rid}")
        names = ["（テーマなし）"] + [p["name"] for p in projects]
        cur = next((p["name"] for p in projects if p["id"] == row["project_id"]), "（テーマなし）")
        st.selectbox("テーマ", names, index=names.index(cur), key=f"rec_e_proj_{rid}")
        st.button("変更を保存", key=f"rec_e_save_{rid}", on_click=_do_update, args=(store, rid, projects))
        st.divider()
        st.caption("計算条件と結果は後から変えられません（記録の信頼性のため）。違う条件は新しく計算して保存してください。")
        confirm = st.checkbox("この記録を削除する（図・添付ファイルも消えます。元に戻せません）", key=f"rec_del_ok_{rid}")
        st.button("🗑 削除", disabled=not confirm, key=f"rec_del_{rid}", on_click=_do_delete, args=(store, rid))


def _render_compare(rows):
    st.subheader(f"{len(rows)} 件を比較")
    modes = {r["mode"] for r in rows}
    qmap = {}
    for m in ("2d", "1d"):
        if m in modes:
            qmap.update(QUANTITIES[m])
    c1, c2 = st.columns(2)
    with c1:
        qname = st.selectbox("比べる量", list(qmap), key="rec_cmp_q")
    all_th = sorted({t for r in rows for t in angles(r)})
    with c2:
        t_sel = (all_th[0] if len(all_th) == 1 else
                 st.select_slider("入射角 θ（無い記録は最も近い角度）", all_th, key="rec_cmp_th"))
    series, skipped = [], []
    for r in rows:
        sp = spectrum(r, qmap[qname], t_sel)
        if sp is None:
            skipped.append(r["title"])
            continue
        wl, y, t_used = sp
        series.append((f"{r['title']}" + ("" if t_used == t_sel else f"（θ={t_used:g}°）"), wl, y))
    st.plotly_chart(spectra_fig(series, f"{qname}, θ={t_sel:g}°"), width="stretch")
    if skipped:
        st.caption(f"この量が無いので表示していない記録: {', '.join(skipped)}")
    st.markdown("**計算条件の違い**（同じ項目は省略）")
    diff = params_diff(rows)
    if diff.empty:
        st.caption("計算条件はすべて同じです。")
    else:
        st.dataframe(diff.astype(str), width="stretch")

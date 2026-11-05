"""解析記録の画面（iridis_app/records_ui.py と 1D/2D 画面の保存欄）の検証。

データベースの代わりにメモリ上の FakeStore（RecordStore と同じメソッド・同じ権限の考え方）を差し込み、
AppTest でボタン操作を再現する。データベース側の権限（RLS）は tests/records_db_check.py で別に確かめる。

 1) 接続先が未設定でもアプリは落ちず、サイドバーに「未設定」と出る。保存欄は出ない
 2) ログイン前は保存欄に「ログインすると保存できます」
 3) サイドバーからログイン → 2D の単一角度を計算 → 保存。計算条件・結果・nk の中身が入る
    計算後に画面の値を変えても、保存されるのは計算した時点の条件
 4) 同じ計算結果は二重に保存されない
 5) 1D の単一角度と角度掃引を保存。保存した値が計算結果と一致する
 6) 「📚 解析記録」: 一覧・1 件の詳細（単一角度/角度掃引）・2 件以上の比較（条件の違いの表）
 7) 自分の記録の題名・タグを変更、削除（図・添付ファイルも消える）。他人の記録には変更欄が出ない
 8) メンバー外の人は保存も閲覧もできない
 3') 保存すると図（結果の PNG・HTML・JSON、構造の HTML・JSON）も付く。詳細画面で表示できる
 9) 「↩ 条件を画面に読み込む」: 2D 画面に切り替わり、NumBasis・波長点数・ピッチ・層が保存時の値に戻る
10) 「↩ 読み込んで再計算」: 2D・1D とも、再計算の結果が保存時の結果と一致すると表示される
12) 画面を使わない部分（restore_2d / restore_1d / compare_results / safe_filename）

実行（WSL、リポジトリ直下で）: source ~/s4env/bin/activate && python3 tests/records_ui_check.py
"""
import copy
import datetime
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
from streamlit.testing.v1 import AppTest

import iridis_app.records_ui as ru
from iridis_app.records import (EDITABLE_FIELDS, LIST_COLUMNS, RecordsError, from_jsonable_array, normalize_tags,
                                safe_filename, to_jsonable)
from iridis_app.records_restore import compare_results, restore_1d, restore_2d

USERS = {"alice@lab.example": ("pw-alice", True), "bob@lab.example": ("pw-bob", True),
         "eve@lab.example": ("pw-eve", False)}


class FakeStore:
    """RecordStore の代わり。全員で 1 つのデータ（db）を共有し、ログイン状態は自分で持つ。"""
    db = {"runs": {}, "projects": {}, "files": {}}

    def upload_file(self, run_id, name, data, content_type, kind="attachment"):
        if self.db["runs"][run_id]["user_id"] != self.user["id"]:
            raise RecordsError(f"{name} のアップロードの権限がありません")
        path = f"{run_id}/{uuid.uuid4().hex[:8]}_{safe_filename(name)}"
        row = {"id": str(uuid.uuid4()), "run_id": run_id, "path": path, "name": name, "kind": kind,
               "content_type": content_type, "size_bytes": len(data), "user_id": self.user["id"],
               "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        self.db["files"][path] = (row, bytes(data))
        return row

    def list_files(self, run_id):
        return [r for r, _ in self.db["files"].values() if r["run_id"] == run_id]

    def download_file(self, path):
        return self.db["files"][path][1]

    def delete_file(self, frow):
        if self.db["runs"][frow["run_id"]]["user_id"] != self.user["id"]:
            raise RecordsError("削除できません（自分の記録のファイルではありません）。")
        del self.db["files"][frow["path"]]

    def __init__(self):
        self.user = None

    def sign_in(self, email, password):
        if USERS.get(email, (None,))[0] != password:
            raise RecordsError("ログインできません: Invalid login credentials")
        self.user = {"id": "uid-" + email.split("@")[0], "email": email}
        return self.user

    def sign_out(self):
        self.user = None

    def change_password(self, new):
        if len(new) < 8:
            raise RecordsError("パスワードは 8 文字以上にしてください。")

    def current_user(self):
        return self.user

    def is_member(self):
        return bool(self.user and USERS[self.user["email"]][1])

    def _need_member(self, action):
        if not self.is_member():
            raise RecordsError(f"{action}の権限がありません")

    def save_run(self, rec):
        self._need_member("保存")
        rid = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.db["runs"][rid] = copy.deepcopy(dict(rec, id=rid, created_at=now, updated_at=now,
                                                  user_id=self.user["id"], owner_email=self.user["email"]))
        return rid

    def list_runs(self, mode=None, project_id=None, tag=None, text=None, user_id=None, limit=200):
        if not self.is_member():
            return []
        rows = [r for r in self.db["runs"].values()
                if (not mode or r["mode"] == mode) and (not project_id or r["project_id"] == project_id)
                and (not tag or tag in r["tags"]) and (not text or text in r["title"])
                and (not user_id or r["user_id"] == user_id)]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return [{k: r.get(k) for k in LIST_COLUMNS.split(",")} for r in rows[:limit]]

    def get_run(self, rid):
        if not self.is_member() or rid not in self.db["runs"]:
            raise RecordsError("記録が見つかりません")
        return copy.deepcopy(self.db["runs"][rid])

    def update_run(self, rid, **fields):
        assert set(fields) <= set(EDITABLE_FIELDS)
        r = self.db["runs"][rid]
        if r["user_id"] != self.user["id"]:
            raise RecordsError("変更できません（自分の記録ではありません）。")
        fields["tags"] = normalize_tags(fields.get("tags", r["tags"]))
        r.update(fields)
        return r

    def delete_run(self, rid):
        if self.db["runs"][rid]["user_id"] != self.user["id"]:
            raise RecordsError("削除できません（自分の記録ではありません）。")
        for f in self.list_files(rid):
            del self.db["files"][f["path"]]
        del self.db["runs"][rid]

    def list_projects(self):
        return list(self.db["projects"].values()) if self.is_member() else []

    def create_project(self, name, description=""):
        p = {"id": str(uuid.uuid4()), "name": name, "description": description, "created_by": self.user["id"]}
        self.db["projects"][p["id"]] = p
        return p


ru.make_store = lambda url, key: FakeStore()


def check(at, label):
    assert not at.exception, f"[{label}] 例外: {[str(e) for e in at.exception]}"
    print(f"[OK] {label}")


def expander(at, label_part):
    return next(e for e in at.expander if label_part in e.label)


def login(at, email):
    at.text_input(key="records_email").input(email)
    at.text_input(key="records_pw").input(USERS[email][0])
    next(b for b in at.button if b.label == "ログイン").click().run()


def save_current(at, title, tags=""):
    cid = next(t.key for t in at.text_input if t.key and t.key.startswith("rec_title_"))[len("rec_title_"):]
    at.text_input(key=f"rec_title_{cid}").input(title)
    at.text_input(key=f"rec_tags_{cid}").input(tags)
    at.button(key=f"rec_save_{cid}").click().run()
    return cid


def wait_2d(at):
    import time
    {b.label: b for b in at.button}["▶ 計算を実行"].click().run()
    for _ in range(600):
        th = at.session_state["calc_state"].get("thread")
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
        at.run()
    at.run()


def main():
    # 1) 未設定（本物の .streamlit/secrets.toml があっても読まないよう、接続先を読む関数を差し替える）
    ru._config = lambda: (None, None)
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=600)
    at.run()
    assert any("未設定です" in c.value for c in at.sidebar.caption)
    at.radio(key="app_mode").set_value("📚 解析記録").run()
    assert any("未設定" in i.value for i in at.info)
    check(at, "[1] 接続先が未設定: サイドバーと解析記録画面に案内が出る")

    ru._config = lambda: ("http://fake", "fake")   # 以降は FakeStore（本物の Supabase にはつながない）
    at = AppTest.from_file(os.path.join(ROOT, "iridis.py"), default_timeout=600)
    at.run()

    # 2) ログイン前の保存欄 / 3) 2D 単一角度を保存
    at.radio(key="app_mode").set_value("2次元周期アレイ (2D)").run()
    {n.label: n for n in at.number_input}["NumBasis（次数）"].set_value(9).run()
    {n.label: n for n in at.number_input}["波長点数"].set_value(3).run()
    wait_2d(at)
    assert any("ログインすると保存できます" in i.value for i in at.info)
    check(at, "[2] ログイン前は保存欄に案内が出る")

    login(at, "bob@lab.example")
    assert any("bob@lab.example" in m.value for m in at.sidebar.markdown)
    pitch_at_calc = at.session_state["calc_state"]["record"]["params"]["sidebar"]["pitch_x_nm"]
    {n.label: n for n in at.number_input}["ピッチ x [nm]"].set_value(pitch_at_calc + 100).run()   # 計算後に画面を変える
    cid = save_current(at, "bob の 2D 円盤", "TiO2, 円盤")
    check(at, "[3] ログインして 2D の結果を保存")
    (rec,) = FakeStore.db["runs"].values()
    cs = at.session_state["calc_state"]
    assert rec["mode"] == "2d" and rec["owner_email"] == "bob@lab.example" and rec["tags"] == ["TiO2", "円盤"]
    assert rec["params"]["sidebar"]["pitch_x_nm"] == pitch_at_calc, "計算した時点の条件が保存されていない"
    assert np.allclose(from_jsonable_array(rec["result"]["R0"]), cs["result"]["R0"], rtol=1e-8)
    assert rec["result"]["n_basis"] == len(cs["result"]["order_basis"]) and rec["params"]["array_layers"]
    assert "SiO2" in rec["nk_snapshot"], list(rec["nk_snapshot"])
    print(f"     保存: R0={np.round(from_jsonable_array(rec['result']['R0']), 4)}  nk={list(rec['nk_snapshot'])}"
          f"  条件={rec['summary']['text'][:60]}…")
    files = {f["name"]: f for f in FakeStore().list_files(rec["id"])}
    expect = {"spectrum.png", "spectrum.html", "spectrum.json", "structure_3d.html", "structure_3d.json",
              "cross_section.html", "cross_section.json"}
    assert expect <= set(files), sorted(files)
    assert all(f["kind"] == "figure" for f in files.values())
    assert FakeStore.db["files"][files["spectrum.png"]["path"]][1].startswith(b"\x89PNG")
    assert b"plotly" in FakeStore.db["files"][files["structure_3d.html"]["path"]][1][:5000].lower()
    print(f"     図 {len(files)} 件: {sorted(files)}")

    # 4) 二重保存しない
    at.run()
    assert any("保存済み" in s.value for s in at.success) and len(FakeStore.db["runs"]) == 1
    check(at, "[4] 同じ計算結果は保存済みと表示され、二重に保存されない")

    # 5) 1D 単一角度・角度掃引
    at.radio(key="app_mode").set_value("列ベース構造 (1D)").run()
    for _ in range(2):
        at.button[0].click().run()
    at.button(key="calc_btn").click().run()
    save_current(at, "bob の 1D 単一角度")
    next(r for r in at.radio if r.label == "Angle mode").set_value("Angle sweep (3D)").run()
    next(s for s in at.slider if s.label == "Angle range [deg]").set_value((0.0, 40.0)).run()
    next(n for n in at.number_input if n.label == "Angle step [deg]").set_value(20.0).run()
    at.button(key="calc_btn").click().run()
    save_current(at, "bob の 1D 角度掃引", "SiO2")
    check(at, "[5] 1D の単一角度・角度掃引を保存")
    r1d = {r["title"]: r for r in FakeStore.db["runs"].values() if r["mode"] == "1d"}
    res = at.session_state["calc_cache"]["result"]
    sw = r1d["bob の 1D 角度掃引"]
    assert np.allclose(from_jsonable_array(sw["result"]["Rp2d"]), res["Rp2d"], rtol=1e-8)
    assert sw["params"]["angle_range"] == [0.0, 40.0] and sw["params"]["columns"]
    single = r1d["bob の 1D 単一角度"]
    wl, y, t = ru.spectrum(single, "Rp")
    irp = from_jsonable_array(single["result"]["irp"])
    assert t == 0.0 and np.allclose(y, irp[:, irp.shape[1] // 2])
    wl, y, t = ru.spectrum(sw, "Rs", 25.0)
    assert t == 20.0 and np.allclose(y, res["Rs2d"][1], rtol=1e-8)
    print(f"     1D: {len(r1d)} 件、spectrum() が次数・最も近い角度を正しく選ぶ")
    names = {f["name"] for f in FakeStore().list_files(sw["id"])}
    assert {"spectrum.png", "heatmap.png", "heatmap.json", "structure.json", "cross_section.json"} <= names, names
    print(f"     1D 角度掃引の図: {sorted(names)}")

    # 他のメンバーの記録を 1 件入れておく
    alice = FakeStore()
    alice.sign_in("alice@lab.example", "pw-alice")
    alice_id = alice.save_run(dict(rec, title="alice の 2D", tags=["alice"]))

    # 6) 解析記録画面
    at.radio(key="app_mode").set_value("📚 解析記録").run()
    df = at.dataframe[0].value
    assert len(df) == 4 and set(df["作成者"]) == {"bob@lab.example", "alice@lab.example"}
    at.text_input(key="rec_f_tag").input("SiO2").run()
    assert list(at.dataframe[0].value["題名"]) == ["bob の 1D 角度掃引"]
    at.text_input(key="rec_f_tag").input("").run()
    at.checkbox(key="rec_f_mine").check().run()
    assert len(at.dataframe[0].value) == 3
    at.checkbox(key="rec_f_mine").uncheck().run()
    check(at, "[6] 一覧と絞り込み（タグ・自分の記録だけ）")

    ids = {r["title"]: r["id"] for r in FakeStore.db["runs"].values()}
    sid = ids["bob の 1D 角度掃引"]
    at.multiselect(key="rec_pick").set_value([sid]).run()
    check(at, "[6] 角度掃引の記録を開く（計算画面と同じ Gsolver'S COLOR.html 互換表示）")
    assert at.radio(key=f"rec_ord_{sid}").value == "全次数合算 (0次光含む)"   # 計算画面と同じ既定
    at.radio(key=f"rec_pol_{sid}").set_value("Rs").run()
    at.radio(key=f"rec_ord_{sid}").set_value("0次光以外 (回折光 m≠0 合算)").run()
    assert not any("簡易ヒートマップ" in i.value for i in at.info)
    check(at, "[6] 偏光・表示する光成分を切り替え（計算画面と同じ選択肢）")
    n_charts = len(at.get("plotly_chart"))
    at.toggle(key=f"rec_showfiles_{sid}").set_value(True).run()
    n_json = sum(1 for f in FakeStore().list_files(sid) if f["name"].endswith(".json"))
    n_png = sum(1 for f in FakeStore().list_files(sid) if f["name"].endswith(".png"))
    assert len(at.get("plotly_chart")) == n_charts + n_json and len(at.get("image")) == n_png, \
        (n_charts, n_json, len(at.get("plotly_chart")), len(at.get("image")), n_png)
    check(at, f"[6] 図・添付ファイルを表示（PNG と、保存した図 {n_json} 枚を描き直す）")
    at.multiselect(key="rec_pick").set_value([ids["bob の 2D 円盤"]]).run()
    check(at, "[6] 2D 単一角度の記録を開く")
    at.multiselect(key="rec_pick").set_value([ids["bob の 2D 円盤"], ids["bob の 1D 単一角度"],
                                              ids["bob の 1D 角度掃引"]]).run()
    check(at, "[6] 3 件（1D と 2D）を比較")
    assert any("この量が無い" in c.value for c in at.caption)   # R0 は 1D の記録には無い
    diff = at.dataframe[-1].value
    assert len(diff) > 0
    print(f"     条件の違いの表: {len(diff)} 項目（例: {list(diff.index[:3])}）")

    # 9) 条件の読み込み（2D）
    rid2d = ids["bob の 2D 円盤"]
    saved2d = FakeStore.db["runs"][rid2d]
    at.multiselect(key="rec_pick").set_value([rid2d]).run()
    at.button(key=f"rec_restore_{rid2d}").click().run()
    check(at, "[9] 2D の記録の条件を画面に読み込む")
    assert at.session_state["app_mode"] == "2次元周期アレイ (2D)"
    assert at.session_state["a2d_tab"] == "🧱 構造"   # 読み込むだけなら構造のタブを開く
    assert any("条件を読み込みました" in i.value for i in at.info)
    nums = {n.label: n.value for n in at.number_input}
    assert nums["NumBasis（次数）"] == 9 and nums["波長点数"] == 3
    assert nums["ピッチ x [nm]"] == saved2d["params"]["sidebar"]["pitch_x_nm"]   # 計算後に変えた値ではなく計算時の値
    strip = lambda ls: [{k: v for k, v in to_jsonable(l).items() if k != "id"} for l in ls]
    assert strip(at.session_state["array_layers"]) == strip(saved2d["params"]["array_layers"])
    print(f"     NumBasis={nums['NumBasis（次数）']}・波長点数={nums['波長点数']}・ピッチ={nums['ピッチ x [nm]']}・層 "
          f"{len(at.session_state['array_layers'])} 枚が戻った")

    # 10) 読み込んで再計算 → 保存時の結果と一致
    at.radio(key="app_mode").set_value("📚 解析記録").run()
    at.multiselect(key="rec_pick").set_value([rid2d]).run()
    at.button(key=f"rec_rerun_{rid2d}").click().run()
    assert at.session_state["a2d_tab"] == "📈 計算結果"   # 再計算なら結果のタブを開く
    for _ in range(600):
        th = at.session_state["calc_state"].get("thread")
        if th is None or not th.is_alive():
            break
        import time
        time.sleep(0.5)
        at.run()
    at.run()
    assert any("同じ結果になりました" in s.value for s in at.success), [s.value for s in at.success]
    check(at, "[10] 2D: 読み込んで再計算 → 保存時の結果と同じ")
    at.radio(key="app_mode").set_value("📚 解析記録").run()
    at.multiselect(key="rec_pick").set_value([sid]).run()
    at.button(key=f"rec_rerun_{sid}").click().run()
    assert at.session_state["app_mode"] == "列ベース構造 (1D)"
    assert next(s for s in at.slider if s.label == "Angle range [deg]").value == (0.0, 40.0)
    assert next(n for n in at.number_input if n.label == "Angle step [deg]").value == 20.0
    assert len(at.session_state["columns"]) == 2
    assert any("同じ結果になりました" in s.value for s in at.success), [s.value for s in at.success]
    check(at, "[10] 1D 角度掃引: 読み込んで再計算 → 保存時の結果と同じ")
    at.radio(key="app_mode").set_value("📚 解析記録").run()

    # 7) 変更・削除・他人の記録
    rid = ids["bob の 2D 円盤"]
    at.multiselect(key="rec_pick").set_value([rid]).run()
    at.text_input(key=f"rec_e_title_{rid}").input("bob の 2D 円盤（確認済み）")
    at.text_input(key=f"rec_e_tags_{rid}").input("TiO2, 確認済み")
    at.button(key=f"rec_e_save_{rid}").click().run()
    assert FakeStore.db["runs"][rid]["title"] == "bob の 2D 円盤（確認済み）"
    assert FakeStore.db["runs"][rid]["tags"] == ["TiO2", "確認済み"]
    at.multiselect(key="rec_pick").set_value([alice_id]).run()
    assert any("他のメンバーの記録" in c.value for c in at.caption)
    assert not any(b.key == f"rec_del_{alice_id}" for b in at.button)
    at.multiselect(key="rec_pick").set_value([rid]).run()
    assert at.button(key=f"rec_del_{rid}").disabled
    at.checkbox(key=f"rec_del_ok_{rid}").check().run()
    at.button(key=f"rec_del_{rid}").click().run()
    assert rid not in FakeStore.db["runs"] and len(at.dataframe[0].value) == 3
    assert not FakeStore().list_files(rid)
    check(at, "[7] 自分の記録の変更・削除（確認のチェックが必要）。他人の記録には変更欄なし")

    # テーマ
    at.text_input(key="rec_new_project").input("TiO2 円盤シリーズ")
    at.button(key="rec_create_project").click().run()
    assert [p["name"] for p in FakeStore.db["projects"].values()] == ["TiO2 円盤シリーズ"]
    check(at, "     テーマを作成")

    # 8) メンバー外
    at.button(key="records_logout").click().run()
    login(at, "eve@lab.example")
    assert any("メンバーに登録されていない" in w.value for w in at.warning)
    check(at, "[8] メンバー外の人には記録が見えない")

    # 12) 画面を使わない部分
    ss = {"_array_layer_id": 5, "lens_a": 999.0, "lens_shape": "cosine"}
    params = {"sidebar": {"lattice_shape": "矩形格子", "pitch_x_nm": 500.0, "pitch_y_nm": 300.0,
                          "env_spec": {"re": 1.33, "im": 0.0}, "subst_spec": "存在しない材質", "angle_mode": "角度掃引",
                          "theta_range": [0.0, 50.0], "n_theta": 6, "phi_deg": 15.0, "pol_deg": 90.0,
                          "wl_min_nm": 400.0, "wl_max_nm": 700.0, "n_freq": 31, "norder": 101},
              "array_layers": [{"id": 1, "type": "disk", "material": "TiO2", "bg_material": "__ENV__",
                                "thickness_nm": 80.0, "radius_nm": 120.0}],
              "step_regions": None, "step_film": [],
              "lens": [{"shape": "paraboloid", "concave": True, "base_radius_nm": 100.0, "height_nm": 50.0,
                        "cx_nm": 0.0, "cy_nm": 0.0, "deposition": "normal", "dz_max_nm": 5.0,
                        "film_extent": "disk", "disk_radius_nm": 100.0},
                       [{"id": 1, "type": "pair", "mat_a": "TiO2", "th_a": 50.0, "mat_b": "無い材質", "th_b": 80.0,
                         "n_pairs": 3}]]}
    warn = restore_2d(ss, params, ["TiO2", "SiO2", "ZrO2"])
    assert ss["pitch_y"] == 300.0 and ss["env_mode"] == "カスタム数値" and ss["env_n_custom"] == 1.33
    assert ss["theta_range"] == (0.0, 50.0) and ss["theta_step"] == 10.0 and ss["norder"] == 101
    assert ss["array_layers"][0]["id"] == 6 and ss["_array_layer_id"] == 6
    assert ss["lens_enabled"] and not ss["step_enabled"] and "lens_a" not in ss and "lens_shape" not in ss
    assert ss["lens_spec"]["shape"] == "paraboloid" and ss["lens_film_layers"][0]["id"] == 1
    assert any("存在しない材質" in w for w in warn) and any("無い材質" in w for w in warn) and len(warn) == 2
    assert "subst_material" not in ss   # 無い材質は書き戻さない（選択欄がエラーになるので）
    ss1 = {"_ncol_id": 3, "_nlayer_id": 7}
    restore_1d(ss1, {"angle_mode": "Single angle (2D)", "inc_angle": 12.0, "wl_option": "Any", "wl_min": 400.0,
                     "wl_max": 600.0, "wl_n": 21, "n_env": 1.0, "substrate": "SiO2", "norder": 11, "repr_order": 9,
                     "columns": [{"id": 1, "width": 200.0, "film_layers": [{"id": 1, "type": "simple"}]}]},
               ["SiO2"])
    assert ss1["wl_range"] == (400.0, 600.0) and ss1["wln"] == 21 and "repr_order" not in ss1   # ±5 次の外
    assert ss1["columns"][0]["id"] == 4 and ss1["columns"][0]["film_layers"][0]["id"] == 8
    print("[12] restore_2d（レンズ・カスタム環境・無い材質の注意）・restore_1d（id の振り直し）")

    a = {"wl_nm": [400.0, 500.0], "R0": [0.1, None], "T0": [0.5, 0.5]}
    c = compare_results(a, {"wl_nm": [400.0, 500.0], "R0": [0.1, None], "T0": [0.5, 0.5000001]})
    assert c["grid_ok"] and c["diffs"]["R0"] == 0.0 and abs(c["max"] - 1e-7) < 1e-12
    assert compare_results(a, {"wl_nm": [400.0, 500.0], "R0": [0.1, 0.2], "T0": [0.5, 0.5]})["max"] == float("inf")
    assert not compare_results(a, {"wl_nm": [400.0, 510.0], "R0": [0.1, None]})["grid_ok"]
    assert safe_filename("スペクトル 図1.PNG") == "1.png" and safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("") == "file" and safe_filename("SEM image (x10k).jpg") == "SEM_image_x10k.jpg"
    print("[12] compare_results（NaN の扱い・点の違い）・safe_filename")
    print("ALL OK")


if __name__ == "__main__":
    main()

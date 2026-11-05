"""解析記録データベース（supabase/schema.sql）と RecordStore（iridis_app/records.py）の検証。

本物の Supabase の代わりに、手元で PostgreSQL 16 と PostgREST 12（Supabase が使っているのと同じ
REST サーバー）を起動し、Supabase の auth.uid() / auth.jwt() とロール（anon / authenticated）を
まねた土台の上で schema.sql を実行する。RecordStore からの呼び出しは supabase-py と同じ
postgrest クライアントで送るので、「どの操作が許され、どれが拒否されるか」をそのまま確かめられる。
（ログイン＝Supabase Auth そのものは使わず、同じ形の JWT をこちらで署名して渡す。）

確かめること:
 1) schema.sql を 2 回実行してもエラーにならない
 2) メンバーは保存でき、作成者・メールはログイン中の人で自動的に埋まる。配列は往復で元に戻る
 3) 他のメンバーの記録は読めるが、変更・削除はできない
 4) 自分の記録でも、計算条件・結果・作成者は後から変えられない（題名・メモ・タグ・テーマだけ）
 5) 他人の名前で保存できない（なりすまし不可）
 6) ログインしていてもメンバー表に無い人は、何も見えず保存もできない
 7) 未ログイン（anon キーだけ）では何も読めない
 8) テーマの作成・共有・重複名・削除（テーマを消しても記録は残る）
 9) 一覧の絞り込み（モード・タグ・題名・作成者・テーマ）
10) updated_at は変更で進み、created_at は変わらない
11) 図・添付ファイル（Storage のバケット run-files と一覧 run_files）: メンバーは読める。付ける・消すのは
    自分の記録だけ（Storage に直接置いても、一覧に直接足しても拒否）。種類・大きさの制限。
    記録を消すと Storage の中身も一覧も消える
（Storage は、本物と同じくログイン中の人の権限で storage.objects を読み書きする代用品 FakeBucket で確かめる。）

必要なもの（WSL）: ~/pgbin（PostgreSQL と postgrest のバイナリ）と ~/pgtest_env（supabase, psycopg,
pyjwt, numpy）。準備の手順は supabase/README.md の「テスト」。
実行: ~/pgtest_env/bin/python3 tests/records_db_check.py
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json

import jwt
import numpy as np
import psycopg
from postgrest import SyncPostgrestClient

from iridis_app.records import RecordStore, RecordsError, from_jsonable_array, make_record

PGBIN = os.environ.get("PGBIN", os.path.expanduser("~/pgbin/bin"))
JWT_SECRET = "test-secret-for-iridis-records-0123456789"

# Supabase の土台をまねる（本物の Supabase では最初から用意されている部分）
BOOTSTRAP_SQL = """
create role anon nologin noinherit;
create role authenticated nologin noinherit;
create role authenticator login noinherit;
grant anon, authenticated to authenticator;
create schema auth;
create table auth.users (id uuid primary key, email text);
create function auth.jwt() returns jsonb language sql stable as $$
  select coalesce(nullif(current_setting('request.jwt.claims', true), ''), '{}')::jsonb $$;
create function auth.uid() returns uuid language sql stable as $$
  select nullif(auth.jwt() ->> 'sub', '')::uuid $$;
grant usage on schema auth to anon, authenticated;
grant usage on schema public to anon, authenticated;
-- Supabase は public の新しい表に anon / authenticated の全権限を既定で付ける（schema.sql 側で絞る）
alter default privileges in schema public grant all on tables to anon, authenticated;
alter default privileges in schema public grant all on functions to anon, authenticated;
-- Supabase Storage の表（本物の一部だけ）。Storage の API はログイン中の人の権限でこの表を読み書きするので、
-- RLS の考え方は同じになる。storage.foldername は Supabase の定義と同じ。
create schema storage;
create table storage.buckets (id text primary key, name text not null, public boolean default false,
  file_size_limit bigint, allowed_mime_types text[], created_at timestamptz default now());
create table storage.objects (id uuid primary key default gen_random_uuid(), bucket_id text references storage.buckets (id),
  name text not null, owner uuid default auth.uid(), created_at timestamptz default now(), metadata jsonb,
  unique (bucket_id, name));
alter table storage.objects enable row level security;
create function storage.foldername(name text) returns text[] language plpgsql as $$
declare _parts text[];
begin
  select string_to_array(name, '/') into _parts;
  return _parts[1:array_length(_parts, 1) - 1];
end $$;
grant usage on schema storage to anon, authenticated;
grant all on storage.objects to anon, authenticated;
grant select on storage.buckets to anon, authenticated;
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def token(user=None):
    now = int(time.time())
    if user is None:
        return jwt.encode({"role": "anon", "iat": now, "exp": now + 3600}, JWT_SECRET, algorithm="HS256")
    return jwt.encode({"role": "authenticated", "aud": "authenticated", "sub": user["id"], "email": user["email"],
                       "iat": now, "exp": now + 3600}, JWT_SECRET, algorithm="HS256")


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeAuth:
    def __init__(self, user):
        self.user = user

    def get_session(self):
        return None if self.user is None else _Obj(user=_Obj(**self.user))


class FakeBucket:
    """storage3 のバケット操作（upload / download / remove）の代わり。Supabase の Storage API と同じく、
    ログイン中の人のロールと JWT で storage.objects を読み書きする（中身は blobs に置く）。
    バケットの大きさ・種類の制限も本物と同じく確かめる。"""
    blobs = {}

    def __init__(self, dsn, bucket, user):
        self.dsn, self.bucket, self.user = dsn, bucket, user

    def _as_user(self, sql, args):
        with psycopg.connect(self.dsn) as db:
            with db.transaction():
                db.execute("set local role " + ("authenticated" if self.user else "anon"))
                claims = ({"role": "authenticated", "sub": self.user["id"], "email": self.user["email"]}
                          if self.user else {"role": "anon"})
                db.execute("select set_config('request.jwt.claims', %s, true)", (json.dumps(claims),))
                return db.execute(sql, args).fetchall()

    def upload(self, path, data, file_options=None):
        ctype = (file_options or {}).get("content-type", "text/plain")
        with psycopg.connect(self.dsn) as db:
            limit, types = db.execute("select file_size_limit, allowed_mime_types from storage.buckets where id = %s",
                                      (self.bucket,)).fetchone()
        if len(data) > limit or ctype not in types:
            raise Exception(f"payload too large or mime type {ctype} not supported")
        self._as_user("insert into storage.objects (bucket_id, name, metadata) values (%s, %s, %s) returning name",
                      (self.bucket, path, json.dumps({"mimetype": ctype, "size": len(data)})))
        FakeBucket.blobs[path] = data
        return {"path": path}

    def download(self, path):
        if not self._as_user("select name from storage.objects where bucket_id = %s and name = %s", (self.bucket, path)):
            raise Exception("Object not found")
        return FakeBucket.blobs[path]

    def remove(self, paths):
        rows = self._as_user("delete from storage.objects where bucket_id = %s and name = any(%s) returning name",
                             (self.bucket, list(paths)))
        for (name,) in rows:
            FakeBucket.blobs.pop(name, None)
        return [{"name": n} for (n,) in rows]


class FakeStorage:
    def __init__(self, dsn, user):
        self.dsn, self.user = dsn, user

    def from_(self, bucket):
        return FakeBucket(self.dsn, bucket, self.user)


class Client:
    """supabase.Client の代わり（.auth / .table / .rpc / .storage）。表は PostgREST に直接つなぐ。"""

    def __init__(self, base_url, user=None, dsn=None):
        self.auth = FakeAuth(user)
        self.storage = FakeStorage(dsn, user)
        self.pg = SyncPostgrestClient(base_url, headers={"Authorization": f"Bearer {token(user)}"})

    def table(self, name):
        return self.pg.from_(name)

    def rpc(self, name, params=None):
        return self.pg.rpc(name, params or {})


def expect_error(fn, words, label):
    try:
        fn()
    except Exception as e:   # RecordsError も、生の postgrest の APIError も
        msg = getattr(e, "message", None) or str(e)
        assert any(w in msg for w in words), (label, msg)
        print(f"    拒否された（期待どおり）: {label} → {msg[:70]}")
        return
    raise AssertionError(f"拒否されるはずが通った: {label}")


def main():
    work = tempfile.mkdtemp(prefix="iridis_pg_")
    pg_port, rest_port = free_port(), free_port()
    procs = []
    try:
        subprocess.run([f"{PGBIN}/initdb", "-D", f"{work}/data", "-U", "postgres", "-A", "trust", "-E", "UTF8",
                        "--locale=C"], check=True, capture_output=True)
        subprocess.run([f"{PGBIN}/pg_ctl", "-D", f"{work}/data", "-l", f"{work}/pg.log", "-w", "start",
                        "-o", f"-p {pg_port} -k {work} -c listen_addresses=127.0.0.1"], check=True, capture_output=True)
        dsn = f"host=127.0.0.1 port={pg_port} user=postgres dbname=postgres"
        schema = open(os.path.join(ROOT, "supabase", "schema.sql"), encoding="utf-8").read()
        with psycopg.connect(dsn, autocommit=True) as db:
            db.execute(BOOTSTRAP_SQL)
            db.execute(schema)
            db.execute(schema)   # 1) 2 回目もエラーにならない
            print("[1] schema.sql を 2 回実行してエラーなし")
            users = {n: {"id": str(uuid.uuid4()), "email": f"{n}@lab.example"} for n in ("alice", "bob", "eve")}
            for u in users.values():
                db.execute("insert into auth.users values (%s, %s)", (u["id"], u["email"]))
            db.execute("insert into public.lab_members (email) values ('alice@lab.example'), ('bob@lab.example')")

        conf = f"{work}/rest.conf"
        with open(conf, "w") as f:
            f.write(f'db-uri = "postgres://authenticator@127.0.0.1:{pg_port}/postgres"\n'
                    f'db-schemas = "public"\ndb-anon-role = "anon"\njwt-secret = "{JWT_SECRET}"\n'
                    f'server-host = "127.0.0.1"\nserver-port = {rest_port}\n')
        procs.append(subprocess.Popen([os.path.join(os.path.dirname(PGBIN), "postgrest"), conf],
                                      stdout=open(f"{work}/rest.log", "w"), stderr=subprocess.STDOUT))
        base = f"http://127.0.0.1:{rest_port}"
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", rest_port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        time.sleep(0.5)

        alice, bob, eve = (RecordStore(Client(base, users[n], dsn)) for n in ("alice", "bob", "eve"))
        anon = RecordStore(Client(base, None, dsn))
        assert alice.is_member() and bob.is_member() and not eve.is_member()

        # 2) 保存と往復
        R0 = np.array([[0.1, 0.2, np.nan], [0.3, 0.123456789123, 0.5]])
        rec = make_record("2d", "TiO2 円盤 r=150", params={"pitch_x_nm": 400.0, "layers": [{"type": "disk"}]},
                          result={"wl_nm": np.array([400.0, 500.0, 600.0]), "R0": R0},
                          summary={"angle_mode": "角度掃引"}, materials=["TiO2", "存在しない材質", 1.5],
                          memo="テスト", tags="TiO2, 円盤, 円盤", elapsed_s=1.5)
        rid = alice.save_run(rec)
        row = bob.get_run(rid)
        assert row["user_id"] == users["alice"]["id"] and row["owner_email"] == "alice@lab.example"
        assert row["tags"] == ["TiO2", "円盤"] and list(row["nk_snapshot"]) == ["TiO2"]
        back = from_jsonable_array(row["result"]["R0"])
        assert np.isnan(back[0, 2]) and abs(back[1, 1] - 0.123456789) < 1e-15 and np.allclose(back[0, :2], [0.1, 0.2])
        print(f"[2] alice が保存 → bob が読める。作成者={row['owner_email']}、NaN と配列が戻る、"
              f"nk の中身 {len(row['nk_snapshot']['TiO2'])} 文字、git={str(row['git_commit'])[:8]}")

        # 3) 他人の記録は変更・削除できない
        expect_error(lambda: bob.update_run(rid, title="乗っ取り"), ["自分の記録ではありません"], "bob が alice の題名を変更")
        expect_error(lambda: bob.delete_run(rid), ["自分の記録ではありません"], "bob が alice の記録を削除")
        assert bob.get_run(rid)["title"] == "TiO2 円盤 r=150"

        # 4) 自分の記録でも計算条件・結果・作成者は変えられない
        created = row["created_at"]
        time.sleep(0.05)
        upd = alice.update_run(rid, title="TiO2 円盤 r=150（確認済み）", tags=["TiO2", "確認済み"], memo="再計算と一致")
        assert upd["title"].endswith("（確認済み）") and upd["created_at"] == created and upd["updated_at"] > row["updated_at"]
        print("[4] alice は自分の記録の題名・タグ・メモを変更できる（updated_at が進み、created_at は同じ）  [10]")
        for col, val in (("params", {}), ("result", {}), ("user_id", users["bob"]["id"]), ("owner_email", "x@y"),
                         ("created_at", "2000-01-01T00:00:00Z")):
            expect_error(lambda: alice.client.table("runs").update({col: val}).eq("id", rid).execute(),
                         ["permission denied"], f"alice が自分の記録の {col} を変更（直接）")
        expect_error(lambda: alice.update_run(rid, params={}), ["変更できない項目"], "update_run で params")

        # 5) なりすまし
        bad = dict(rec, user_id=users["bob"]["id"])
        expect_error(lambda: alice.client.table("runs").insert(bad).execute(), ["row-level security"],
                     "alice が bob の名前で保存")
        bad = dict(rec, owner_email="bob@lab.example")
        expect_error(lambda: alice.client.table("runs").insert(bad).execute(), ["row-level security"],
                     "alice が bob のメールで保存")

        # 6) メンバーでない人
        assert eve.list_runs() == [] and eve.list_projects() == []
        expect_error(lambda: eve.save_run(rec), ["権限がありません"], "メンバー外の eve が保存")
        expect_error(lambda: eve.get_run(rid), ["見つかりません"], "メンバー外の eve が記録を開く")
        expect_error(lambda: eve.client.table("lab_members").insert({"email": "eve@lab.example"}).execute(),
                     ["permission denied"], "eve が自分をメンバー表に追加")
        print("[6] メンバー外の eve には一覧が空")

        # 7) 未ログイン
        for t in ("runs", "projects", "lab_members"):
            expect_error(lambda: anon.client.table(t).select("*").execute(), ["permission denied"], f"未ログインで {t} を読む")
        expect_error(lambda: anon.is_member(), ["権限がありません"], "未ログインで is_lab_member")

        # 8) テーマ
        prj = alice.create_project("構造色 TiO2 円盤", "2026 秋")
        assert [p["name"] for p in bob.list_projects()] == ["構造色 TiO2 円盤"]
        expect_error(lambda: bob.create_project("構造色 TiO2 円盤"), ["同じ名前"], "同じ名前のテーマ")
        res = bob.client.table("projects").update({"name": "改名"}).eq("id", prj["id"]).execute()
        assert res.data == [] and bob.list_projects()[0]["name"] == "構造色 TiO2 円盤"
        print("    拒否された（期待どおり）: bob が alice のテーマを改名（0 行）")
        rid1d = bob.save_run(make_record("1d", "1D 格子", params={"pitch_nm": 500}, result={"irp": [[0.1]]},
                                         tags=["SiO2"], project_id=prj["id"]))
        alice.update_run(rid, project_id=prj["id"])

        # 9) 絞り込み
        def ids(**kw):
            return {r["id"] for r in alice.list_runs(**kw)}
        assert ids() == {rid, rid1d}
        assert ids(mode="1d") == {rid1d} and ids(mode="2d") == {rid}
        assert ids(tag="確認済み") == {rid} and ids(tag="SiO2") == {rid1d} and ids(tag="無い") == set()
        assert ids(text="円盤") == {rid} and ids(text="格子") == {rid1d}
        assert ids(user_id=users["bob"]["id"]) == {rid1d} and ids(project_id=prj["id"]) == {rid, rid1d}
        assert "result" not in alice.list_runs()[0]   # 一覧では重い列を取らない
        print("[9] 絞り込み（モード・タグ・題名・作成者・テーマ）が正しい。一覧に result は含まれない")

        # 8 続き) テーマを消しても記録は残る
        alice.client.table("projects").delete().eq("id", prj["id"]).execute()
        assert alice.get_run(rid)["project_id"] is None and bob.get_run(rid1d)["project_id"] is None
        print("[8] テーマ: 作成・共有・重複名の拒否・他人による改名の拒否・削除しても記録は残る")

        # 11) 図・添付ファイル（Storage と run_files）
        png = b"\x89PNG\r\n\x1a\n" + b"0" * 100
        f1 = alice.upload_file(rid, "スペクトル 図1.png", png, "image/png", kind="figure")
        f2 = alice.upload_file(rid, "SEM像.jpg", b"jpegdata", "image/jpeg")
        assert f1["path"].startswith(f"{rid}/") and f1["path"].isascii() and f1["path"].endswith(".png")
        assert f1["name"] == "スペクトル 図1.png" and f1["user_id"] == users["alice"]["id"]
        assert [f["name"] for f in bob.list_files(rid)] == ["スペクトル 図1.png", "SEM像.jpg"]
        assert bob.download_file(f1["path"]) == png
        print(f"[11] alice が図と添付を付け、bob が読める（Storage のキー: {f1['path'].split('/', 1)[1]}）")
        expect_error(lambda: bob.upload_file(rid, "x.png", png, "image/png"), ["権限がありません"],
                     "bob が alice の記録にファイルを付ける")
        expect_error(lambda: bob.client.storage.from_("run-files").upload(f"{rid}/evil.png", png, {"content-type": "image/png"}),
                     ["row-level security"], "bob が alice の記録のフォルダへ Storage に直接置く")
        expect_error(lambda: bob.client.table("run_files").insert({
            "run_id": rid, "path": f"{rid}/x.png", "name": "x", "kind": "figure", "content_type": "image/png",
            "size_bytes": 1}).execute(), ["row-level security"], "bob が alice の記録のファイル一覧に行を足す")
        expect_error(lambda: bob.delete_file(f1), ["自分の記録のファイルではありません"], "bob が alice の図を消す")
        expect_error(lambda: alice.client.table("run_files").update({"name": "改名"}).eq("id", f1["id"]).execute(),
                     ["permission denied"], "ファイル一覧の書き換え（直接）")
        expect_error(lambda: alice.client.table("run_files").insert({
            "run_id": rid, "path": f"{rid1d}/x.png", "name": "x", "kind": "figure", "content_type": "image/png",
            "size_bytes": 1}).execute(), ["run_files_path_in_run", "row-level security"],
            "alice が別の記録のパスを自分の記録の一覧に載せる")
        expect_error(lambda: eve.download_file(f1["path"]), ["ファイルの取得に失敗"], "メンバー外の eve がダウンロード")
        expect_error(lambda: anon.download_file(f1["path"]), ["ファイルの取得に失敗"], "未ログインでダウンロード")
        expect_error(lambda: alice.upload_file(rid, "a.exe", b"MZ", "application/x-msdownload"), ["この種類のファイル"],
                     "許していない種類のファイル")
        expect_error(lambda: alice.client.storage.from_("run-files").upload(f"{rid}/a.exe", b"MZ",
                                                                           {"content-type": "application/x-msdownload"}),
                     ["not supported"], "許していない種類を Storage に直接置く（バケットの制限）")
        alice.delete_file(f2)
        assert [f["name"] for f in alice.list_files(rid)] == ["スペクトル 図1.png"] and f2["path"] not in FakeBucket.blobs
        print("[11] 自分のファイルは消せる（Storage と一覧の両方から消える）")

        # 削除（図・添付ファイルも消える）
        expect_error(lambda: bob.delete_run(rid), ["自分の記録ではありません"], "bob が alice の記録をファイルごと削除")
        assert f1["path"] in FakeBucket.blobs
        alice.delete_run(rid)
        assert ids() == {rid1d} and not [p for p in FakeBucket.blobs if p.startswith(rid)]
        with psycopg.connect(dsn) as db:
            assert db.execute("select count(*) from public.run_files where run_id = %s", (rid,)).fetchone()[0] == 0
            assert db.execute("select count(*) from storage.objects where name like %s", (rid + "/%",)).fetchone()[0] == 0
        print("[3] 自分の記録は削除できる（Storage の中身・ファイル一覧も消える）")
        print("ALL OK")
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)
        subprocess.run([f"{PGBIN}/pg_ctl", "-D", f"{work}/data", "-m", "fast", "stop"], capture_output=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

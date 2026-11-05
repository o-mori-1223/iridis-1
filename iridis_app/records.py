"""解析記録の保存・読み出し（Supabase）。画面（Streamlit）には依存しない部分。

- `RecordStore`: Supabase クライアントを包み、ログイン・記録の保存/一覧/取得/編集/削除・テーマの
  一覧/作成・図や添付ファイルの付け外し（Storage のバケット run-files と一覧 run_files）を行う。**1 ブラウザセッションに 1 つ**作る（クライアントがログイン状態を持つので、
  st.cache_resource で全員に共有してはいけない）。
- `to_jsonable` / `from_jsonable_array`: numpy 配列・複素数・NaN を JSON に入れられる形へ変換する。
- `nk_snapshot` / `git_info`: 再現用に、使った nk ファイルの中身とコードの版を集める。

権限はデータベース側（supabase/schema.sql の RLS）で守る。ここでのチェックは使いやすさのためだけ。
"""
import math
import mimetypes
import os
import re
import subprocess
import threading
import uuid

import numpy as np

SCHEMA_VERSION = 1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_NK_DIR = os.path.join(ROOT, "data", "nk")

BUCKET = "run-files"
MAX_FILE_BYTES = 20 * 2**20   # schema.sql のバケット設定と同じ
ALLOWED_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml", "application/pdf",
                 "text/html", "application/json", "text/csv", "text/plain")


def safe_filename(name):
    """Storage のキーに使える名前（英数字と . _ -）にする。拡張子は残す。"""
    base, ext = os.path.splitext(os.path.basename(name or ""))
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")[:80] or "file"
    ext = re.sub(r"[^A-Za-z0-9]+", "", ext)[:10]
    return f"{base}.{ext.lower()}" if ext else base


def guess_type(name, given=None):
    """アップロードされたファイルの種類（ブラウザが送る値が無ければ拡張子から）。"""
    if given and given in ALLOWED_TYPES:
        return given
    t = mimetypes.guess_type(name)[0]
    return t or given or "application/octet-stream"


# 一覧で取る列（params / result / nk_snapshot は重いので詳細を開いたときだけ取る）
LIST_COLUMNS = "id,created_at,updated_at,user_id,owner_email,project_id,mode,title,memo,tags,summary,git_commit,git_dirty"
EDITABLE_FIELDS = ("title", "memo", "tags", "project_id")


class RecordsError(Exception):
    """利用者に見せてよい日本語のエラー。"""


# ---------------------------------------------------------------------------
# JSON 変換
# ---------------------------------------------------------------------------
def _num(v, digits):
    if isinstance(v, complex):
        return {"re": _num(v.real, digits), "im": _num(v.imag, digits)}
    v = float(v)
    if not math.isfinite(v):
        return None   # JSON に NaN は入らない。読み出し時に NaN へ戻す
    return float(f"{v:.{digits}g}") if digits else v


def to_jsonable(x, digits=None):
    """numpy 配列・数値・複素数・tuple を含む値を JSON に入れられる形にする。
    digits を指定すると、配列の中の実数を有効数字 digits 桁に丸める（結果の容量を約半分にする。
    反射率なら 9 桁で十分）。関数など変換できない値は TypeError（保存漏れに気づけるように）。"""
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist(), digits)
    if isinstance(x, np.generic):
        x = x.item()
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, (float, complex)):
        return _num(x, digits)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, digits) for v in x]
    raise TypeError(f"JSON に保存できない値です: {type(x).__name__}")


def from_jsonable_array(x):
    """to_jsonable で保存した数値の入れ子リストを numpy 配列へ戻す（None → NaN、{"re","im"} → 複素数）。"""
    def conv(v):
        if v is None:
            return math.nan
        if isinstance(v, dict) and "re" in v:
            return complex(conv(v["re"]), conv(v["im"]))
        if isinstance(v, list):
            return [conv(e) for e in v]
        return v
    return np.array(conv(x))


# ---------------------------------------------------------------------------
# 再現用の情報
# ---------------------------------------------------------------------------
def nk_snapshot(names, nk_dir=_DEFAULT_NK_DIR):
    """材質名のうち data/nk/<名前>.nk があるものについて、ファイルの中身をそのまま返す。
    nk ファイルは手元で書き換えられるので、名前だけでは後から同じ計算を再現できない。"""
    out = {}
    for name in sorted({str(n) for n in names if isinstance(n, str)}):
        path = os.path.join(nk_dir, name + ".nk")
        if os.path.isfile(path):
            with open(path, encoding="utf-8-sig") as f:
                out[name] = f.read()
    return out


def git_info(cwd=ROOT):
    """(commit, dirty)。git が使えなければ (None, None)。dirty は追跡中の .py に未コミットの変更があるか
    （eigen 同梱ファイルの改行など計算に関係ない差分は見ない）。OneDrive 上（WSL から /mnt/c）だと
    status に数秒〜数十秒かかるので、start_git_info() で起動時に一度だけ裏で取る。"""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                                timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "*.py"], cwd=cwd,
                                capture_output=True, text=True, timeout=120, check=True).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


# 動いているコードは起動時に読み込んだものなので、版も起動時に 1 回取れば足りる。
_git_state = {"thread": None, "value": (None, None)}
_git_lock = threading.Lock()


def start_git_info():
    """git_info() を裏のスレッドで 1 回だけ実行し始める（何度呼んでもよい）。"""
    with _git_lock:
        if _git_state["thread"] is None:
            def run():
                _git_state["value"] = git_info()
            _git_state["thread"] = threading.Thread(target=run, daemon=True)
            _git_state["thread"].start()


def cached_git_info(wait_s=30.0):
    """start_git_info() の結果。まだ終わっていなければ最大 wait_s 秒待ち、それでもなければ (None, None)。"""
    start_git_info()
    _git_state["thread"].join(wait_s)
    return _git_state["value"]


def make_record(mode, title, params, result, summary=None, materials=(), memo="", tags=(),
                project_id=None, elapsed_s=None, nk_dir=_DEFAULT_NK_DIR):
    """runs テーブルに入れる 1 行を作る。user_id・owner_email はデータベースがログイン中の人で埋める。"""
    if mode not in ("1d", "2d"):
        raise ValueError(f"mode は '1d' か '2d': {mode!r}")
    title = (title or "").strip()
    if not title:
        raise RecordsError("題名を入力してください。")
    commit, dirty = cached_git_info()
    return {
        "mode": mode,
        "title": title[:200],
        "memo": memo or "",
        "tags": normalize_tags(tags),
        "project_id": project_id,
        "schema_version": SCHEMA_VERSION,
        "params": to_jsonable(params),
        "summary": to_jsonable(summary or {}),
        "result": to_jsonable(result, digits=9),
        "nk_snapshot": nk_snapshot(materials, nk_dir),
        "git_commit": commit,
        "git_dirty": dirty,
        "elapsed_s": None if elapsed_s is None else float(elapsed_s),
    }


def normalize_tags(tags):
    """'TiO2, 円盤 ,円盤' のような文字列やリストを、空白を除いた重複なしのリストにする。"""
    if isinstance(tags, str):
        tags = tags.replace("、", ",").split(",")
    return list(dict.fromkeys(t.strip() for t in tags if t and t.strip()))


# ---------------------------------------------------------------------------
# Supabase とのやり取り
# ---------------------------------------------------------------------------
def _api_error(e, action):
    """action は「保存」「テーマの作成」のような名詞。"""
    msg = getattr(e, "message", None) or str(e)
    code = getattr(e, "code", None)
    if code == "42501" or "row-level security" in msg or "permission denied" in msg:
        return RecordsError(f"{action}の権限がありません（研究室メンバーに登録されているか、"
                            f"自分の記録かを確認してください）。")
    if code == "23505":
        return RecordsError(f"{action}ができません（同じ名前がすでにあります）。")
    if "runs_result_size" in msg:
        return RecordsError(f"{action}ができません（結果が大きすぎます。波長点数・角度点数を減らしてください）。")
    return RecordsError(f"{action}に失敗しました: {msg}")


class RecordStore:
    """Supabase クライアント 1 つ（= ログイン状態 1 つ）を包む。

    client は supabase.Client と同じく .auth / .table(name) / .rpc(name) を持つもの。
    テストでは PostgREST に直接つないだ代用品を渡す（tests/records_check.py）。"""

    def __init__(self, client):
        self.client = client

    @classmethod
    def connect(cls, url, anon_key):
        from supabase import create_client
        return cls(create_client(url, anon_key))

    # ---- ログイン ----
    def sign_in(self, email, password):
        from supabase_auth.errors import AuthError
        try:
            res = self.client.auth.sign_in_with_password({"email": email.strip().lower(), "password": password})
        except AuthError as e:
            raise RecordsError(f"ログインできません: {e.message}") from e
        return {"id": res.user.id, "email": res.user.email}

    def sign_out(self):
        try:
            self.client.auth.sign_out()
        except Exception:   # ネットワーク切れでも手元のログイン状態は消す
            pass

    def change_password(self, new_password):
        from supabase_auth.errors import AuthError
        if len(new_password) < 8:
            raise RecordsError("パスワードは 8 文字以上にしてください。")
        try:
            self.client.auth.update_user({"password": new_password})
        except AuthError as e:
            raise RecordsError(f"パスワードを変更できません: {e.message}") from e

    def current_user(self):
        """ログイン中なら {"id", "email"}。トークンの期限切れはここで自動更新される。"""
        try:
            session = self.client.auth.get_session()
        except Exception:
            return None
        if session is None or session.user is None:
            return None
        return {"id": session.user.id, "email": session.user.email}

    def is_member(self):
        try:
            return bool(self.client.rpc("is_lab_member").execute().data)
        except Exception as e:
            raise _api_error(e, "メンバーの確認") from e

    # ---- 記録 ----
    def save_run(self, record):
        try:
            res = self.client.table("runs").insert(record).execute()
        except Exception as e:
            raise _api_error(e, "保存") from e
        if not res.data:
            raise RecordsError("保存の権限がありません（研究室メンバーに登録されているか確認してください）。")
        return res.data[0]["id"]

    def list_runs(self, mode=None, project_id=None, tag=None, text=None, user_id=None, limit=200):
        q = self.client.table("runs").select(LIST_COLUMNS)
        if mode:
            q = q.eq("mode", mode)
        if project_id:
            q = q.eq("project_id", project_id)
        if tag:
            q = q.contains("tags", [tag])
        if text:
            q = q.ilike("title", f"%{text}%")
        if user_id:
            q = q.eq("user_id", user_id)
        try:
            return q.order("created_at", desc=True).limit(limit).execute().data
        except Exception as e:
            raise _api_error(e, "一覧の取得") from e

    def get_run(self, run_id):
        try:
            rows = self.client.table("runs").select("*").eq("id", run_id).limit(1).execute().data
        except Exception as e:
            raise _api_error(e, "記録の取得") from e
        if not rows:
            raise RecordsError("記録が見つかりません（消されたか、見る権限がありません）。")
        return rows[0]

    def update_run(self, run_id, **fields):
        bad = set(fields) - set(EDITABLE_FIELDS)
        if bad:
            raise RecordsError(f"変更できない項目です: {', '.join(sorted(bad))}（計算条件と結果は後から変えられません）")
        if "tags" in fields:
            fields["tags"] = normalize_tags(fields["tags"])
        if "title" in fields and not (fields["title"] or "").strip():
            raise RecordsError("題名を入力してください。")
        try:
            res = self.client.table("runs").update(fields).eq("id", run_id).execute()
        except Exception as e:
            raise _api_error(e, "変更") from e
        if not res.data:
            raise RecordsError("変更できません（自分の記録ではありません）。")
        return res.data[0]

    def delete_run(self, run_id):
        """記録を消す。図・添付ファイルの中身（Storage）を先に消す（一覧 run_files は記録と一緒に消える）。"""
        row = self.get_run(run_id)
        me = self.current_user()
        if me is None or row.get("user_id") != me["id"]:
            raise RecordsError("削除できません（自分の記録ではありません）。")
        paths = [f["path"] for f in self.list_files(run_id)]
        if paths:
            try:
                removed = self._bucket().remove(paths)
            except Exception as e:
                raise _api_error(e, "図・添付ファイルの削除") from e
            if len(removed or []) < len(paths):
                raise RecordsError("図・添付ファイルを消せなかったため、記録は消していません。")
        try:
            res = self.client.table("runs").delete().eq("id", run_id).execute()
        except Exception as e:
            raise _api_error(e, "削除") from e
        if not res.data:
            raise RecordsError("削除できません（自分の記録ではありません）。")

    # ---- 図・添付ファイル（Storage）----
    def _bucket(self):
        return self.client.storage.from_(BUCKET)

    def upload_file(self, run_id, name, data, content_type, kind="attachment"):
        """記録 run_id に 1 ファイルを付ける。Storage のキーは英数字に直した名前（日本語のキーは使えない）、
        元の名前は一覧 run_files.name に残す。戻り値は run_files の行。"""
        if kind not in ("figure", "attachment"):
            raise ValueError(kind)
        if content_type not in ALLOWED_TYPES:
            raise RecordsError(f"{name}: この種類のファイルは付けられません（{content_type}）。"
                               f"画像・PDF・CSV・テキスト・JSON・HTML だけです。")
        if len(data) > MAX_FILE_BYTES:
            raise RecordsError(f"{name}: 大きすぎます（{len(data) / 1e6:.1f} MB。上限 {MAX_FILE_BYTES // 2**20} MB）。")
        path = f"{run_id}/{uuid.uuid4().hex[:8]}_{safe_filename(name)}"
        try:
            self._bucket().upload(path, data, {"content-type": content_type, "upsert": "false"})
        except Exception as e:
            raise _api_error(e, f"{name} のアップロード") from e
        try:
            res = self.client.table("run_files").insert({
                "run_id": run_id, "path": path, "name": name[:200], "kind": kind,
                "content_type": content_type, "size_bytes": len(data)}).execute()
        except Exception as e:
            try:
                self._bucket().remove([path])   # 一覧に載らなかった中身を残さない
            except Exception:
                pass
            raise _api_error(e, f"{name} の登録") from e
        return res.data[0]

    def list_files(self, run_id):
        try:
            return (self.client.table("run_files").select("*").eq("run_id", run_id)
                    .order("created_at").execute().data)
        except Exception as e:
            raise _api_error(e, "ファイル一覧の取得") from e

    def download_file(self, path):
        try:
            return self._bucket().download(path)
        except Exception as e:
            raise _api_error(e, "ファイルの取得") from e

    def delete_file(self, file_row):
        try:
            removed = self._bucket().remove([file_row["path"]])
        except Exception as e:
            raise _api_error(e, "ファイルの削除") from e
        if not removed:
            raise RecordsError("削除できません（自分の記録のファイルではありません）。")
        try:
            self.client.table("run_files").delete().eq("id", file_row["id"]).execute()
        except Exception as e:
            raise _api_error(e, "ファイルの削除") from e

    # ---- テーマ ----
    def list_projects(self):
        try:
            return self.client.table("projects").select("id,name,description,created_by").order("name").execute().data
        except Exception as e:
            raise _api_error(e, "テーマ一覧の取得") from e

    def create_project(self, name, description=""):
        name = (name or "").strip()
        if not name:
            raise RecordsError("テーマ名を入力してください。")
        try:
            res = self.client.table("projects").insert({"name": name, "description": description}).execute()
        except Exception as e:
            raise _api_error(e, "テーマの作成") from e
        return res.data[0]

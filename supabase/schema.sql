-- Iridis 解析記録データベース（Supabase / PostgreSQL）
--
-- 使い方: Supabase の管理画面 → SQL Editor にこのファイルの中身を貼って Run。
-- 何度実行しても同じ状態になる（作り直しは不要）。手順の全体は supabase/README.md。
--
-- しくみ:
--   - lab_members  研究室メンバーのメールアドレス一覧。ここに載っている人だけが記録を読み書きできる
--                  （ログインできても、載っていなければ何も見えない）。管理者が SQL Editor で追加する。
--   - projects     記録をまとめるテーマ（試料・研究テーマなど）。
--   - runs         解析記録 1 件 = 1 行。計算条件（params）・結果（result）・使った nk ファイルの中身
--                  （nk_snapshot）・コードの版（git_commit）を丸ごと残す。
--   - run_files    記録に付けた図・添付ファイルの一覧。中身は Storage のバケット run-files の
--                  <記録 ID>/<ファイル名> に置く。記録を消すと一覧も消える（Storage の中身はアプリが先に消す）。
--   権限（RLS = 行単位のアクセス制御）:
--     読む   メンバーなら全員の記録を読める
--     作る   メンバーが自分の名前でのみ作れる（他人になりすませない）
--     直す   自分の記録の 題名・メモ・タグ・テーマ だけ（計算条件と結果は後から変えられない）
--     消す   自分の記録だけ

-- ------------------------------------------------------------------
-- 研究室メンバー
-- ------------------------------------------------------------------
create table if not exists public.lab_members (
  email        text primary key check (email = lower(email)),
  display_name text not null default '',
  added_at     timestamptz not null default now()
);
alter table public.lab_members enable row level security;

-- ログイン中の人がメンバーか。security definer なので lab_members の RLS に関係なく引ける。
create or replace function public.is_lab_member() returns boolean
language sql stable security definer set search_path = ''
as $$
  select exists (
    select 1 from public.lab_members m
    where m.email = lower(coalesce(auth.jwt() ->> 'email', ''))
  );
$$;
revoke all on function public.is_lab_member() from public, anon;
grant execute on function public.is_lab_member() to authenticated;

revoke all on public.lab_members from anon, authenticated;
grant select on public.lab_members to authenticated;
drop policy if exists lab_members_select on public.lab_members;
create policy lab_members_select on public.lab_members
  for select to authenticated using (public.is_lab_member());

-- ------------------------------------------------------------------
-- テーマ（記録のまとまり）
-- ------------------------------------------------------------------
create table if not exists public.projects (
  id          uuid primary key default gen_random_uuid(),
  created_at  timestamptz not null default now(),
  name        text not null unique check (char_length(name) between 1 and 100),
  description text not null default '',
  created_by  uuid default auth.uid() references auth.users (id) on delete set null
);
alter table public.projects enable row level security;

revoke all on public.projects from anon, authenticated;
grant select, insert, delete on public.projects to authenticated;
grant update (name, description) on public.projects to authenticated;

drop policy if exists projects_select on public.projects;
create policy projects_select on public.projects
  for select to authenticated using (public.is_lab_member());
drop policy if exists projects_insert on public.projects;
create policy projects_insert on public.projects
  for insert to authenticated with check (public.is_lab_member() and created_by = auth.uid());
drop policy if exists projects_update on public.projects;
create policy projects_update on public.projects
  for update to authenticated using (public.is_lab_member() and created_by = auth.uid());
drop policy if exists projects_delete on public.projects;
create policy projects_delete on public.projects
  for delete to authenticated using (public.is_lab_member() and created_by = auth.uid());

-- ------------------------------------------------------------------
-- 解析記録
-- ------------------------------------------------------------------
create table if not exists public.runs (
  id             uuid primary key default gen_random_uuid(),
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now(),
  -- 作った人。ユーザーを消しても記録は残す（user_id だけ空になる）。
  user_id        uuid default auth.uid() references auth.users (id) on delete set null,
  owner_email    text not null default lower(coalesce(auth.jwt() ->> 'email', '')),
  project_id     uuid references public.projects (id) on delete set null,
  mode           text not null check (mode in ('1d', '2d')),
  title          text not null check (char_length(title) between 1 and 200),
  memo           text not null default '',
  tags           text[] not null default '{}',
  schema_version integer not null default 1,
  params         jsonb not null,                 -- 計算条件（構造・入射条件・NumBasis など）
  summary        jsonb not null default '{}',    -- 一覧表示用の要約（軽い）
  result         jsonb not null,                 -- 計算結果（波長・角度・R/T の配列）
  nk_snapshot    jsonb not null default '{}',    -- 使った nk ファイルの中身 {材質名: テキスト}
  git_commit     text,
  git_dirty      boolean,
  elapsed_s      double precision,
  -- 無料枠（DB 500 MB）を 1 件で食いつぶさないための上限（約 8 MB）
  constraint runs_result_size check (pg_column_size(result) < 8000000)
);
create index if not exists runs_created_at_idx on public.runs (created_at desc);
create index if not exists runs_project_idx on public.runs (project_id);
create index if not exists runs_user_idx on public.runs (user_id);
create index if not exists runs_tags_idx on public.runs using gin (tags);

create or replace function public.runs_touch() returns trigger
language plpgsql set search_path = ''
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;
drop trigger if exists runs_touch on public.runs;
create trigger runs_touch before update on public.runs
  for each row execute function public.runs_touch();

alter table public.runs enable row level security;

-- 列単位の権限: 後から変えられるのは 題名・メモ・タグ・テーマ だけ。
revoke all on public.runs from anon, authenticated;
grant select, insert, delete on public.runs to authenticated;
grant update (title, memo, tags, project_id) on public.runs to authenticated;

drop policy if exists runs_select on public.runs;
create policy runs_select on public.runs
  for select to authenticated using (public.is_lab_member());
drop policy if exists runs_insert on public.runs;
create policy runs_insert on public.runs
  for insert to authenticated with check (
    public.is_lab_member()
    and user_id = auth.uid()
    and owner_email = lower(coalesce(auth.jwt() ->> 'email', ''))
  );
drop policy if exists runs_update on public.runs;
create policy runs_update on public.runs
  for update to authenticated using (public.is_lab_member() and user_id = auth.uid());
drop policy if exists runs_delete on public.runs;
create policy runs_delete on public.runs
  for delete to authenticated using (public.is_lab_member() and user_id = auth.uid());

-- ------------------------------------------------------------------
-- 図・添付ファイル（中身は Storage のバケット run-files、ここはその一覧）
-- ------------------------------------------------------------------
-- 記録 run_id（文字列）がログイン中の人のものか。Storage の権限（パスの先頭 = 記録 ID）でも使う。
create or replace function public.owns_run(run_id text) returns boolean
language sql stable security definer set search_path = ''
as $$
  select exists (select 1 from public.runs r where r.id::text = run_id and r.user_id = auth.uid());
$$;
revoke all on function public.owns_run(text) from public, anon;
grant execute on function public.owns_run(text) to authenticated;

create table if not exists public.run_files (
  id           uuid primary key default gen_random_uuid(),
  run_id       uuid not null references public.runs (id) on delete cascade,
  created_at   timestamptz not null default now(),
  user_id      uuid default auth.uid() references auth.users (id) on delete set null,
  path         text not null unique,                       -- Storage のキー: <記録 ID>/<ファイル名>
  name         text not null check (char_length(name) between 1 and 200),   -- 表示名（元のファイル名）
  kind         text not null check (kind in ('figure', 'attachment')),       -- figure = 自動で作った図
  content_type text not null,
  size_bytes   bigint not null check (size_bytes >= 0),
  constraint run_files_path_in_run check (path like run_id::text || '/%')
);
create index if not exists run_files_run_idx on public.run_files (run_id);
alter table public.run_files enable row level security;

revoke all on public.run_files from anon, authenticated;
grant select, insert, delete on public.run_files to authenticated;

drop policy if exists run_files_select on public.run_files;
create policy run_files_select on public.run_files
  for select to authenticated using (public.is_lab_member());
drop policy if exists run_files_insert on public.run_files;
create policy run_files_insert on public.run_files
  for insert to authenticated with check (
    public.is_lab_member() and user_id = auth.uid() and public.owns_run(run_id::text)
  );
drop policy if exists run_files_delete on public.run_files;
create policy run_files_delete on public.run_files
  for delete to authenticated using (public.is_lab_member() and public.owns_run(run_id::text));

-- Storage のバケット（非公開。1 ファイル 20 MB まで、種類も限定）
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('run-files', 'run-files', false, 20971520,
        array['image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml', 'application/pdf',
              'text/html', 'application/json', 'text/csv', 'text/plain'])
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit,
                               allowed_mime_types = excluded.allowed_mime_types;

-- Storage の権限: メンバーは読める。置く・消すのは、パスの先頭の記録が自分のものの場合だけ。
-- 上書き（update）は許さない。
drop policy if exists "run-files select" on storage.objects;
create policy "run-files select" on storage.objects
  for select to authenticated using (bucket_id = 'run-files' and public.is_lab_member());
drop policy if exists "run-files insert" on storage.objects;
create policy "run-files insert" on storage.objects
  for insert to authenticated with check (
    bucket_id = 'run-files' and public.is_lab_member() and public.owns_run((storage.foldername(name))[1])
  );
drop policy if exists "run-files delete" on storage.objects;
create policy "run-files delete" on storage.objects
  for delete to authenticated using (
    bucket_id = 'run-files' and public.is_lab_member() and public.owns_run((storage.foldername(name))[1])
  );

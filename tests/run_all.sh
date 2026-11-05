#!/bin/bash
# tests/ 以下の確認スクリプトをまとめて実行する（WSL）。
#   bash tests/run_all.sh
set -e
source ~/s4env/bin/activate
cd "$(dirname "$0")/.."
for t in s4_parallel_check.py nan_regression_check.py hemisphere_check.py curved_multilayer_check.py cross_section_check.py hcylinder_check.py convergence_check.py polarization_check.py cylinder_cut_check.py app_smoke_1d.py app_smoke_2d.py cylinder_cut_ui_check.py records_ui_check.py; do
  echo "=== $t"
  python3 "tests/$t" 2>&1 | grep -v -E 'use_container_width|will be removed|width=|^$|st\.iframe|components\.v1|ScriptRunContext|No runtime found|without `streamlit run`'
done
# 解析記録データベースの権限（RLS）テストは、手元の PostgreSQL・PostgREST と別の venv を使う
# （用意の手順は supabase/README.md の「テスト」）。用意していなければ飛ばす。
echo "=== records_db_check.py"
if [ -x ~/pgbin/bin/postgres ] && [ -x ~/pgbin/postgrest ] && [ -x ~/pgtest_env/bin/python3 ]; then
  ~/pgtest_env/bin/python3 tests/records_db_check.py
else
  echo "（飛ばしました: ~/pgbin と ~/pgtest_env がありません。supabase/README.md の「テスト」参照）"
fi

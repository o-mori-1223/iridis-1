"""Iridis（iridis.py）のUI部品。

- common.py          : 両モード共通（nk読み込み、CSV、固定表示の計算ボタン、S4未導入時の停止）
- column_editor.py   : 1D 列ビルダー（列・膜層の編集UI）
- column_figures.py  : 1D 構造図・断面図・層テーブル
- column_calc.py     : 1D 計算（S4、角度ごとのキャッシュ＋プロセス並列）
- column_app.py      : 1D モード画面（render_column_app）
- array2d_model.py   : 2D 層構成 → S4 layers 変換、格子ベクトル等の純粋関数
- array2d_editor.py  : 2D 層構成・基板段差の編集UI
- array2d_preview.py : 2D 3D構造プレビュー
- array2d_app.py     : 2D モード画面（render_array2d_app）
- array2d_supercell_editor.py : 2D「周期領域と円柱の切り抜き」の設定欄（計算は RCWA2D/cylinder_cut.py）
- records.py         : 解析記録（Supabase）の保存・読み出し。画面に依存しない（RecordStore, to_jsonable 等）
- records_ui.py      : 解析記録の画面（サイドバーのログイン、保存欄、「📚 解析記録」画面）
- records_figures.py : 解析記録の図（記録からスペクトルを取り出す、PNG / HTML / JSON を作る）
- records_restore.py : 保存した条件を画面の入力欄へ戻す、再計算の結果を保存時と比べる

計算本体は RCWA2D/（rcwa_s4_backend, rcwa_s4_array2d_backend, s4_parallel, s4lab）。
"""

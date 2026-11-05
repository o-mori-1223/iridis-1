# Structural Color Simulator 🦋

RCWA（厳密結合波解析）を用いて、多層膜・回折格子構造による**構造色**をシミュレーションする Streamlit アプリです。

光の入射条件や層構成を指定すると、反射スペクトルを計算し、対応する色を表示します。

サイドバーの **Angle mode** で2つの表示を切り替えられます。

- **Single angle (2D)**: 入射角を1つ固定し、横軸=波長・縦軸=反射率の2次元グラフを表示。
- **Angle sweep (3D)**: 角度の範囲と刻みを指定し、波長×角度×反射率の3Dサーフェスを表示（偏光 Rp/Rs を切替可能）。色計算・CSV出力は代表角度（スイープ先頭）のスペクトルに基づきます。

## 必要環境

- Python 3.14
- 主な依存ライブラリ
  - streamlit
  - numpy / pandas / scipy
  - plotly
  - colour-science

## セットアップ

仮想環境（`.venv`）が同梱されています。そのまま使う場合はインストール不要です。

新規に環境を作る場合:

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit numpy pandas scipy plotly colour-science
```

## 起動方法

```bash
cd /Users/sotaro711/lab/koumyou_sim
source .venv/bin/activate
streamlit run app.py
```

仮想環境を有効化せずに直接起動することもできます:

```bash
.venv/bin/streamlit run app.py
```

起動後、自動でブラウザが開き `http://localhost:8501` でアクセスできます。

## ファイル構成

| ファイル / ディレクトリ | 説明 |
| --- | --- |
| `app.py` | メインアプリ（構造色シミュレーター本体） |
| `rcwa_mh.py` | RCWA 計算モジュール（`app.py` から import） |
| `ppap.py` | アプリの派生版 |
| `app,org.py` | 旧版・バックアップ |
| `color2.py` | スペクトル → 色変換の補助スクリプト |
| `data/nk/` | 各材料の屈折率データ（`.nk` ファイル） |
| `ColourExcel/` | 色計算の検証用スクリプトとデータ |

## 材料データについて

`data/nk/` には材料ごとの複素屈折率データ（波長, n, k）が `.nk` 形式で格納されています。
新しい材料を追加する場合は、同じフォーマットのファイルをこのディレクトリに置くとアプリの選択肢に表示されます。

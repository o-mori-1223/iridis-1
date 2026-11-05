"""Iridis — 構造色シミュレータ（起動: `bash run_koumyou.sh`、WSL）。

サイドバー上部の「構造タイプ」で以下を切り替える:
- 列ベース構造 (1D): 周期を列に分割し、各列で基板の上げ高さとフィルム層を独立に設定する
  （iridis_app/column_app.py）。
- 2次元周期アレイ (2D): 円盤/球を正方・矩形・六方格子上に周期配置する
  （iridis_app/array2d_app.py）。
- 📚 解析記録: 保存した計算結果の一覧・比較（Supabase、iridis_app/records_ui.py、
  設定は supabase/README.md）。

計算エンジンは1D/2Dとも S4（third_party/S4 を RCWA2D/libs4lab.so にビルドし、
RCWA2D/s4lab.py の ctypes ラッパー経由で使う。Linux/WSL専用）。各(波長, 角度)点は
RCWA2D/s4_parallel.py が常駐ワーカープロセスで並列に解く。UI部品の構成は
iridis_app/__init__.py を参照。

（旧ファイル名: R8_koumyou_v5.py → 2026-09-29に iridis.py へ改名。2026-10-26に
iridis_app/ パッケージへ分割）

UIは main() の中で組み、`if __name__ == '__main__'` で呼ぶ（Streamlit はスクリプトを
__main__ として実行する）。multiprocessing の spawn ワーカーは親の __main__ を
__mp_main__ として読み込み直すので、ガードが無いとワーカー内でUIが走ってしまう。
"""
import streamlit as st

from iridis_app.array2d_app import render_array2d_app
from iridis_app.column_app import render_column_app
from iridis_app.common import compact_page_css
from iridis_app.log import get_logger
from iridis_app.records_ui import render_auth_sidebar, render_records_page
from RCWA2D import s4_parallel


@st.cache_resource
def _start_workers():
    """S4並列計算のワーカープロセスを、アプリを開いた時点で裏で起動しておく（1プロセスにつき1回）。"""
    s4_parallel.warm_up()
    mem = s4_parallel._mem_available_bytes()
    get_logger().info("Iridis 起動: S4 ワーカー %d・空きメモリ %s", s4_parallel.n_workers(),
                      f"{mem / 1e9:.1f} GB" if mem else "不明")
    return True


def main():
    st.set_page_config(
        page_title="Iridis",
        page_icon="🦋",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://koumyou.org/',
            'Report a bug': "https://koumyou.org/",
            'About': "# Iridis — structural color simulator"
        }
    )
    compact_page_css()
    app_mode = st.sidebar.radio(
        '構造タイプ',
        ['列ベース構造 (1D)', '2次元周期アレイ (2D)', '📚 解析記録'],
        key='app_mode',
        help='列ベース構造(1D): 周期を列に分割し、各列に基板の上げと薄膜を積む構造。'
             '2次元周期アレイ(2D): 円盤/球を正方・矩形・六方格子上に周期配置する構造。'
             'どちらもS4で計算（WSL側の環境が必要）。'
             '解析記録: 保存した計算結果の一覧・比較（Supabase、ログインが必要）。',
    )
    render_auth_sidebar()
    st.sidebar.divider()
    _start_workers()

    if app_mode == '列ベース構造 (1D)':
        render_column_app()
    elif app_mode == '2次元周期アレイ (2D)':
        render_array2d_app()
    else:
        render_records_page()


if __name__ == '__main__':
    main()

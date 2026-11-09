"""サーバー（起動したターミナル）に出すログ。

計算の開始・進み具合・終了・エラー、解析記録の保存などを、時刻付きで標準エラー出力に書く。
RCWA2D 側（s4_parallel など）は logging.getLogger("iridis.s4") に書けば、ここで付けた出力先に流れる。

環境変数:
- IRIDIS_LOG_LEVEL: DEBUG / INFO（既定）/ WARNING
- IRIDIS_LOG_FILE: ファイルにも書く（例: logs/iridis.log。追記）
"""
import logging
import os
import sys
import threading

_FORMAT = "%(asctime)s [%(name)s] %(levelname)s %(message)s"


def get_logger(name="iridis"):
    """name は "iridis" か "iridis.xxx"。出力先は最初の 1 回だけ付ける（Streamlit は画面のスクリプトを
    何度も実行し直すが、モジュールは 1 回しか読み込まないので、ここで重複しないようにしておけばよい）。"""
    root = logging.getLogger("iridis")
    if not root.handlers:
        fmt = logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S")
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        root.addHandler(h)
        path = os.environ.get("IRIDIS_LOG_FILE")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        root.setLevel(os.environ.get("IRIDIS_LOG_LEVEL", "INFO").upper())
        root.propagate = False   # Streamlit のログ設定に左右されない
    return logging.getLogger(name)


def heartbeat(log, label, progress_fn, interval_s=300.0):
    """計算中、interval_s ごとに「計算中 done/total 点・経過」を書く別スレッドを始め、止めるための Event を返す。
    進み具合の行（ProgressLog）は計算の塊が終わったときにしか書けないので、1 点に何分もかかる計算でも
    動いていることがターミナルで分かるようにする。progress_fn() は (done, total, elapsed_s) を返す。"""
    stop = threading.Event()

    def run():
        while not stop.wait(interval_s):
            done, total, elapsed = progress_fn()
            log.info("%s: 計算中 %d/%d 点・経過 %.0f 秒", label, done, total, elapsed)

    threading.Thread(target=run, daemon=True, name="iridis-heartbeat").start()
    return stop


def fmt_bytes(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class ProgressLog:
    """進み具合を、10% 進んで前の行から 5 秒以上たったときだけ 1 行書く（塊が終わるたびに書くと多すぎる。
    数秒で終わる計算では書かない。終わりは呼び出し側が「完了」を書き、途中の生存確認は heartbeat が書く）。"""

    def __init__(self, log, label, step=0.1, min_gap_s=5.0):
        self.log, self.label, self.step, self.min_gap_s = log, label, step, min_gap_s
        self.next_frac, self.last_t = step, 0.0

    def __call__(self, done, total, elapsed):
        frac = done / total if total else 1.0
        if done == total:
            return
        if frac >= self.next_frac and elapsed - self.last_t >= self.min_gap_s:
            rest = elapsed / done * (total - done) if done else float("nan")
            self.log.info("%s: %d/%d 点（%.0f%%）経過 %.0f 秒・残り約 %.0f 秒",
                          self.label, done, total, 100 * frac, elapsed, rest)
            while self.next_frac <= frac:
                self.next_frac += self.step
            self.last_t = elapsed

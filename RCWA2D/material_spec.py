"""材質指定（環境・基板・薄膜のいずれも）を `wl_nm -> complex` の関数に解決する
共通ヘルパー。

R8_koumyou_v3.py/v4.py の `make_nk_fn()` と同じ規約（数値ならそのまま定数の
n+ik、文字列なら`data/nk/<name>.nk`を表引き）を単一値向けに切り出したもの。
`rcwa_s4_array2d_backend.py`（2D周期アレイ用S4バックエンド）の
`nk_env_fn`/`nk_subst_fn`/`nk_fn_map`はいずれも「波長依存関数」を要求する
規約なので、基板や環境を**数値でそのまま指定したい**場合（例: 基板n=1.5固定、
data/nk/にファイルを作らずに済ませたい）に、このヘルパーで数値→関数へ変換
する。

koumyouアプリ側の`make_nk_fn()`とロジックを重複させたくなかったため、
今後v3/v4アプリ側もこちらを呼ぶように統一する余地があるが、今回は
既存アプリには手を入れず（動作確認済みのコードを不必要に変更しない）、
新規の2Dアレイ用コードだけがこのモジュールを使う。
"""
import os

import numpy as np
from scipy.interpolate import interp1d

_DEFAULT_NK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nk")


def list_nk_materials(nk_dir=None):
    """`data/nk/`にある材質名（拡張子なし）の一覧をソートして返す。
    R8_koumyou_v4.pyの`get_nk_list()`と同じ内容（UIのdropdown用）。
    """
    nk_dir = nk_dir or _DEFAULT_NK_DIR
    names = [os.path.splitext(f)[0] for f in os.listdir(nk_dir)
              if os.path.isfile(os.path.join(nk_dir, f))]
    names.sort()
    return names


def resolve_nk_fn(spec, nk_dir=None):
    """`spec`を`wl_nm -> complex`の関数に解決する。

    spec が数値・複素数（例: 1.5, 1.71+2.88j）なら、波長に関わらず一定の
    n+ikを返す定数関数にする。
    spec が数値の文字列（例: "1.5"）でも同様（float()でそのまま解釈できる
    値。koumyouアプリの`.isnumeric()`判定と違い、小数点も負符号も許容する
    ——"1.5"は`.isnumeric()`ではFalseになってしまうため、より緩い判定に
    している）。
    それ以外の文字列は材質名とみなし、`data/nk/<spec>.nk`
    （形式: 波長(nm) n k、`;`はコメント行）を線形補間して表引きする関数
    にする。
    """
    if isinstance(spec, (int, float, complex)):
        val = complex(spec)
        return lambda wl, _v=val: _v

    if isinstance(spec, str):
        try:
            val = complex(spec)
        except ValueError:
            pass
        else:
            return lambda wl, _v=val: _v

        nk_dir = nk_dir or _DEFAULT_NK_DIR
        fpath = os.path.join(nk_dir, spec + ".nk")
        d = np.loadtxt(fpath, comments=";", encoding="utf-8_sig")
        fn = interp1d(d[:, 0], d[:, 1] + d[:, 2] * 1j, kind="linear", fill_value="extrapolate")
        return lambda wl, _fn=fn: complex(_fn(wl))

    raise TypeError(f"resolve_nk_fn: 数値・複素数・文字列のいずれでもない spec={spec!r}")


if __name__ == '__main__':
    print("=== 数値指定 ===")
    fn_n = resolve_nk_fn(1.5)
    print(f"  resolve_nk_fn(1.5)(550) = {fn_n(550.0)}")
    fn_nk = resolve_nk_fn(1.71 + 2.88j)
    print(f"  resolve_nk_fn(1.71+2.88j)(550) = {fn_nk(550.0)}")
    print(f"  resolve_nk_fn('1.5')(550) = {resolve_nk_fn('1.5')(550.0)}  (数値文字列)")

    print("=== ファイル指定 ===")
    fn_sio2 = resolve_nk_fn('SiO2')
    fn_air = resolve_nk_fn('Air')
    print(f"  resolve_nk_fn('SiO2')(550) = {fn_sio2(550.0)}")
    print(f"  resolve_nk_fn('Air')(550) = {fn_air(550.0)}")

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
import re

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


def parse_nk_text(text):
    """n,k の表（1 列目 = 波長、2 列目 = n、3 列目 = k。k は無くてもよい）→ [N, 3] の配列（波長は nm、昇順）。
    区切りは空白・タブ・カンマ・セミコロン。';' や '#' で始まる行と、数値で始まらない行（見出しなど）は飛ばす。
    波長の最大が 20 未満なら µm とみなして 1000 倍する。"""
    rows = []
    for line in text.lstrip('\ufeff').splitlines():
        line = line.strip()
        if not line or line[0] in ';#"':
            continue
        try:
            v = [float(t) for t in re.split(r'[\s,;]+', line) if t]
        except ValueError:
            continue
        if len(v) >= 2:
            rows.append((v[0], v[1], v[2] if len(v) > 2 else 0.0))
    if len(rows) < 2:
        raise ValueError('数値の行が 2 行未満です（1 列目 = 波長、2 列目 = n、3 列目 = k）。')
    d = np.array(rows, float)
    if not np.all(np.isfinite(d)):
        raise ValueError('数値でない値（nan・inf）があります。')
    if d[:, 0].max() < 20:
        d[:, 0] *= 1000.0
    d = d[np.argsort(d[:, 0], kind='stable')]
    if np.any(np.diff(d[:, 0]) <= 0):
        raise ValueError('同じ波長の行が 2 つ以上あります。')
    return d


def read_nk_file(path):
    with open(path, encoding='utf-8-sig') as f:
        return parse_nk_text(f.read())


def save_nk(name, d, source='', nk_dir=None):
    """n,k の表を data/nk/<name>.nk として保存する。同じ名前があれば上書きせずに ValueError。"""
    name = name.strip()
    if not re.fullmatch(r'[A-Za-z0-9_\-]{1,40}', name) or re.fullmatch(r'[0-9_\-]+', name):
        raise ValueError('名前は半角の英数字・_・- で 40 文字まで（数字だけの名前は不可）にしてください。')
    nk_dir = nk_dir or _DEFAULT_NK_DIR
    if name.lower() in (n.lower() for n in list_nk_materials(nk_dir)):
        raise ValueError(f'「{name}」という材料は既にあります。別の名前にしてください。')
    if np.any(d[:, 1] <= 0) or np.any(d[:, 2] < 0):
        raise ValueError('n は正、k は 0 以上にしてください（吸収は k > 0 で表す）。')
    path = os.path.join(nk_dir, name + '.nk')
    with open(path, 'x', encoding='utf-8', newline='\n') as f:
        f.write('; wavelength(nm) n(real) k(imag)\n')
        if source.strip():
            f.write('; Source: ' + ' '.join(source.split()) + '\n')
        for w, n, k in d:
            f.write(f'{w:.6g} {n:.8g} {k:.8g}\n')
    return path


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
        d = read_nk_file(fpath)
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

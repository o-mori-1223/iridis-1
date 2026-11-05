"""
C++ 版 rcwa_mh モジュールのビルドスクリプト。
koumyou_sim フォルダで以下を実行してください:

    .\.venv\Scripts\python.exe setup_rcwa.py build_ext --inplace

ビルドが成功すると rcwa_mh.cpXXX-win_amd64.pyd が生成されます。
Python は .pyd を .py より優先してインポートするため、
既存の rcwa_mh.py は自動的に C++ 版に切り替わります。
"""

from setuptools import setup, Extension
import pybind11
import os

# Eigen のヘッダディレクトリ（koumyou_sim/eigen-3.4.0 に解凍してください）
EIGEN_DIR = os.path.join(os.path.dirname(__file__), "eigen-3.4.0")

ext = Extension(
    name="rcwa_mh",
    sources=["rcwa_mh_cpp.cpp"],
    include_dirs=[
        pybind11.get_include(),
        EIGEN_DIR,
    ],
    extra_compile_args=[
        "/O2",
        "/std:c++17",
        "/EHsc",
        "/utf-8",    # C4819 warning fix (treat source as UTF-8)
    ],
    language="c++",
)

setup(
    name="rcwa_mh",
    version="1.0",
    ext_modules=[ext],
)

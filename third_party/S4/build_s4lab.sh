#!/bin/bash
# third_party/S4（本家 victorliu/S4 のソース）＋ s4lab_shim.c から、
# RCWA2D/libs4lab.so（RCWA2D/s4lab.py が ctypes で読み込む共有ライブラリ）をビルドする。
#
# WSL(Ubuntu)で実行:  bash third_party/S4/build_s4lab.sh
# 必要パッケージ:     sudo apt install build-essential liblapack-dev libblas-dev libfftw3-dev
#
# 本家の Makefile/Makefile.common はLua/Pythonフロントエンド込みのため使わず、
# Makefile.common の S4_LIBOBJS と同じソース一覧をここで直接コンパイルする。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../../RCWA2D/libs4lab.so"
BUILD="${S4LAB_BUILD_DIR:-$HOME/.cache/s4lab_build}"   # OneDrive上を避けてWSL側でビルド
mkdir -p "$BUILD"

CPPFLAGS="-I$HERE/S4 -I$HERE/S4/RNP -I$HERE/S4/kiss_fft -DHAVE_BLAS -DHAVE_LAPACK -DHAVE_LIBFFTW3"
CFLAGS="-O3 -fPIC -Wall -Wno-unused -Wno-sign-compare"
CXXFLAGS="$CFLAGS -fno-exceptions"
LIBS="-llapack -lblas -lfftw3 -lstdc++ -lm"

CXX_SRCS="S4.cpp rcwa.cpp fmm/fmm_common.cpp fmm/fmm_FFT.cpp fmm/fmm_kottke.cpp fmm/fmm_closed.cpp
          fmm/fmm_PolBasisNV.cpp fmm/fmm_PolBasisVL.cpp fmm/fmm_PolBasisJones.cpp
          fmm/fmm_experimental.cpp fmm/fft_iface.cpp"
C_SRCS="pattern/pattern.c pattern/intersection.c pattern/predicates.c numalloc.c gsel.c sort.c
        kiss_fft/kiss_fft.c kiss_fft/tools/kiss_fftnd.c SpectrumSampler.c cubature.c
        Interpolator.c convert.c"

OBJS=""
for f in $CXX_SRCS; do
  o="$BUILD/$(echo "$f" | tr '/' '_').o"
  g++ -c $CXXFLAGS $CPPFLAGS "$HERE/S4/$f" -o "$o" &
  OBJS="$OBJS $o"
done
for f in $C_SRCS; do
  o="$BUILD/$(echo "$f" | tr '/' '_').o"
  gcc -c $CFLAGS $CPPFLAGS "$HERE/S4/$f" -o "$o" &
  OBJS="$OBJS $o"
done
gcc -c $CFLAGS $CPPFLAGS -std=gnu99 "$HERE/s4lab_shim.c" -o "$BUILD/s4lab_shim.o" &
wait
OBJS="$OBJS $BUILD/s4lab_shim.o"

g++ -shared -o "$OUT" $OBJS $LIBS
echo "built: $(readlink -f "$OUT")"

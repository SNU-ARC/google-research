# python3 configure.py --no-deps
PY3="$(which python3)" && PYTHON_BIN_PATH=$PY3 CC=clang-8 bazel-2.2.0 build -c opt --copt=-mavx2 --copt=-mfma --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w :build_pip_pkg
PYTHON=python3 ./bazel-bin/build_pip_pkg && pip3 install *.whl --force-reinstall

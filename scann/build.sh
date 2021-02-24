# Install bazel
# sudo apt-get install -y software-properties-common curl gnupg rsync
# sudo curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
# sudo echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# sudo apt-get update && sudo apt-get install -y bazel-2.2.0

# Install g++-9
# sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
# sudo apt-get update
# sudo apt-get install -y g++-9 clang-8

# pip3 install --upgrade pip
# git clone https://github.com/google-research/google-research.git --depth=1

# Install ScaNN
# python3 configure.py --no-deps
PY3="$(which python3)" && PYTHON_BIN_PATH=$PY3 CC=clang-8 bazel-2.2.0 build -c opt --copt=-mavx2 --copt=-mfma --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w :build_pip_pkg
PYTHON=python3 ./bazel-bin/build_pip_pkg && pip3 install *.whl --force-reinstall
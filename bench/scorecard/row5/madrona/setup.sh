#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Madrona-MJX + MuJoCo Playground setup on Ubuntu 22.04, NVIDIA driver 580, CUDA 12.9 toolkit (about 40 min with the build).
# Pins: madrona_mjx 1505699168fd, its madrona submodule b46e6ab7, mujoco_playground f2159f3 (the last commit before Madrona
# was replaced by the MJX Warp renderer), jax 0.5.3 (README: < 0.6), mujoco / mujoco-mjx 3.4.0, brax 0.12.5.
# Two fixes are needed on Blackwell with CUDA 12.9: the build must use -DLOAD_VULKAN=ON (an OFF build makes the render GPU
# state empty and the rasterizer segfaults), and sort_archetype.cpp must call cub::ThreadReduce (sed below).
set -euxo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=$HOME/work; mkdir -p $W; cd $W
sudo -n apt-get update -q
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential git cmake ninja-build \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev libvulkan-dev xorg-dev \
  libvulkan1 vulkan-tools libegl1 libgl1
# NVIDIA GL/EGL/Vulkan userspace (some cloud images ship the driver without it): needed by the rasterizer
bash "$HERE/../maniskill/nvidia_egl.sh"
vulkaninfo --summary | grep -q "NVIDIA RTX"
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 $W/venv
. $W/venv/bin/activate
uv pip install "jax[cuda12]==0.5.3" "numpy<2.3" cmake==3.31.6 "mujoco==3.4.0" "mujoco-mjx==3.4.0" "brax==0.12.5" \
  ml_collections etils pillow tqdm opencv-python-headless mediapy absl-py lxml "warp-lang>=1.11"
git clone https://github.com/shacklettbp/madrona_mjx $W/madrona_mjx
git -C $W/madrona_mjx checkout 1505699168fd7c0c12629356f357b05491ce9d0c
git -C $W/madrona_mjx submodule update --init --recursive
git clone https://github.com/google-deepmind/mujoco_playground $W/mp
git -C $W/mp checkout f2159f3
git -C $W/mp submodule update --init --recursive || true   # mujoco_menagerie: the Franka assets
export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH
# CUDA 12.9 CUB: internal::ThreadReduce(array) no longer exists; Madrona's JIT-compiled sort_archetype.cpp needs cub::
sed -i 's/internal::ThreadReduce(warp_histograms/cub::ThreadReduce(warp_histograms/' $W/madrona_mjx/external/madrona/src/mw/device/sort_archetype.cpp
mkdir -p $W/madrona_mjx/build; cd $W/madrona_mjx/build
cmake .. -DLOAD_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
cd .. && uv pip install -e .
uv pip install -e $W/mp --no-deps
python -c "import madrona_mjx, jax; print(jax.__version__, jax.devices())"

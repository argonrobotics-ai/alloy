#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ManiSkill3 setup on Ubuntu 22.04 with an NVIDIA driver 580 (GCP deep-learning image, cu129): NVIDIA EGL/Vulkan userspace,
# ManiSkill@62ff3a58 editable, uv venv ~/ms (Python 3.12). Run once: bash setup.sh
set -euxo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git curl libvulkan1 vulkan-tools libegl1 libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 >/dev/null
[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || bash "$HERE/nvidia_egl.sh"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv python install 3.12
mkdir -p ~/ManiSkill && cd ~/ManiSkill && git init -q && git remote add origin https://github.com/haosulab/ManiSkill
git fetch -q --depth 1 origin 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3 && git checkout -q FETCH_HEAD
git rev-parse HEAD
uv venv --clear ~/ms --python 3.12
uv pip install -p ~/ms/bin/python -e ~/ManiSkill torch pynvml tyro pandas pillow
~/ms/bin/python -c "import mani_skill, sapien, torch; print('MS', mani_skill.__version__, 'sapien', sapien.__version__, 'torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo SETUP_DONE

#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Set up the alloy.create 3D generator (photo/text -> mesh) on a Linux machine with an NVIDIA GPU (16 GB or more),
# in its own folder, and write a script that serves it. The GPU machine does not need Alloy's physics engine: the
# server imports only alloy.create from this checkout, NumPy and the models' packages.
#
#   bash examples/create/generator_setup.sh [FOLDER]        # default ~/alloy-generator; about 10 GB of packages
#   ALLOY_GEN3D_TOKEN=<secret> FOLDER/serve.sh              # serves on port 8765; models download on first use
#
# Then, on the machine that runs alloy.create: export ALLOY_GEN3D_URL=http://<gpu-machine>:8765 and the same
# ALLOY_GEN3D_TOKEN. Models (all allow commercial use): Kandinsky 2.2 (Apache-2.0), rembg U^2-Net (MIT / Apache-2.0),
# TripoSR (MIT), scikit-image marching cubes (BSD-3-Clause).
set -euo pipefail
DIR=$(mkdir -p "${1:-$HOME/alloy-generator}" && cd "${1:-$HOME/alloy-generator}" && pwd)
ALLOY_SRC=$(cd "$(dirname "$0")/../.." && pwd)
TRIPOSR_COMMIT=107cefdc244c39106fa830359024f6a2f1c78871     # alloy.create.gen3d_local.TRIPOSR_COMMIT
PY=${PYTHON:-python3}

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is needed"'
command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver found (nvidia-smi); the generator needs a GPU" >&2; exit 1; }
[ -d "$DIR/venv" ] || "$PY" -m venv "$DIR/venv"
. "$DIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch torchvision
pip install -q diffusers transformers accelerate sentencepiece protobuf "huggingface_hub<2" rembg onnxruntime \
    scikit-image einops omegaconf trimesh pillow imageio numpy
[ -d "$DIR/TripoSR" ] || git clone -q https://github.com/VAST-AI-Research/TripoSR "$DIR/TripoSR"
git -C "$DIR/TripoSR" checkout -q "$TRIPOSR_COMMIT"
python -c 'import torch; assert torch.cuda.is_available(), "PyTorch sees no CUDA GPU"'

cat > "$DIR/serve.sh" <<EOT
#!/bin/bash
# Serve the alloy.create generator. Set ALLOY_GEN3D_TOKEN to require it; --host 127.0.0.1 to keep it local.
. "$DIR/venv/bin/activate"
export ALLOY_TRIPOSR_DIR="$DIR/TripoSR" PYTHONPATH="$ALLOY_SRC\${PYTHONPATH:+:\$PYTHONPATH}" HF_HOME="\${HF_HOME:-$DIR/hf}"
exec python -m alloy.create.serve --host "\${HOST:-0.0.0.0}" --port "\${PORT:-8765}" "\$@"
EOT
chmod +x "$DIR/serve.sh"
PYTHONPATH="$ALLOY_SRC" ALLOY_TRIPOSR_DIR="$DIR/TripoSR" python -c \
    'from alloy.create.gen3d_local import available; assert available(), "the generator packages do not import"'
echo "ready: ALLOY_GEN3D_TOKEN=<secret> $DIR/serve.sh   (then on your machine: export ALLOY_GEN3D_URL=http://<this machine>:8765)"

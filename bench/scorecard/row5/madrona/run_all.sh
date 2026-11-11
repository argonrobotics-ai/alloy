#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One run command after setup.sh: every cell as its own process, alone on the GPU. Resumes: a cell with a .done file is skipped.
# usage: bash run_all.sh [raytracer|rasterizer ...]   (default: raytracer then rasterizer)
# Rows -> ~/work/out/madrona_g4.jsonl, logs -> ~/work/out/logs, PNG grids -> ~/work/out/png. SCENES="cartpole" limits the scenes.
set -u; unset MADRONA_MWGPU_KERNEL_CACHE  # it names a cache FILE; a directory segfaults Madrona
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=$HOME/work; . $W/venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p $W/out/png $W/out/logs
OUT=$W/out/madrona_g4.jsonl
BACKENDS=("$@"); [ ${#BACKENDS[@]} -gt 0 ] || BACKENDS=(raytracer rasterizer)
for be in "${BACKENDS[@]}"; do
 for scene in ${SCENES:-cartpole franka_cube}; do
  for size in 64 128 256; do
   for worlds in 256 1024 4096; do
    tag=${scene}_${be}_${size}_${worlds}
    if [ -f $W/out/logs/$tag.done ]; then continue; fi
    python "$HERE/bench.py" --scene $scene --backend $be --size $size --worlds $worlds --out $OUT \
      --png $W/out/png/$tag.png > $W/out/logs/$tag.log 2>&1
    echo "exit $?" >> $W/out/logs/$tag.log; touch $W/out/logs/$tag.done
   done
  done
 done
done
touch $W/out/ALL_DONE

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DLVM common-cu129-ubuntu-2204-nvidia-580 ships the driver WITHOUT its GL/EGL userspace (only Mesa's EGL vendor file),
# so EGL falls back to Mesa and fails (eglInitialize 0x3001, /dev/dri permission). Install the matching NVIDIA EGL/GL
# libraries from the same-version runfile (extract only; the kernel module is not touched).
set -euo pipefail
V=$(cat /sys/module/nvidia/version)
cd /tmp
[ -f NV.run ] || curl -fsSL -o NV.run https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run \
  || curl -fsSL -o NV.run https://download.nvidia.com/XFree86/Linux-x86_64/$V/NVIDIA-Linux-x86_64-$V.run
rm -rf nvx && sh NV.run --extract-only --target nvx >/dev/null
L=/usr/lib/x86_64-linux-gnu
for f in libEGL_nvidia libGLX_nvidia libGLESv2_nvidia libGLESv1_CM_nvidia libnvidia-eglcore libnvidia-glcore \
         libnvidia-glsi libnvidia-tls libnvidia-gpucomp libnvidia-egl-gbm libnvidia-glvkspirv libnvidia-rtcore \
         libnvidia-allocator libnvoptix libnvidia-ngx; do
  for so in nvx/$f.so.$V; do [ -f "$so" ] && sudo cp "$so" $L/; done
done
cd $L && for f in libEGL_nvidia libGLX_nvidia libGLESv2_nvidia libGLESv1_CM_nvidia; do
  [ -f $f.so.$V ] && sudo ln -sf $f.so.$V $f.so.0; done
# OptiX + NGX (RTX renderers: "Failed to create an Optix adaptor" / "NGX isn't enabled" without them)
for f in libnvoptix libnvidia-ngx; do [ -f $f.so.$V ] && sudo ln -sf $f.so.$V $f.so.1; done
sudo mkdir -p /usr/share/glvnd/egl_vendor.d
sudo cp /tmp/nvx/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
[ -f /tmp/nvx/nvidia_icd.json ] && sudo mkdir -p /usr/share/vulkan/icd.d && sudo cp /tmp/nvx/nvidia_icd.json /usr/share/vulkan/icd.d/
sudo ldconfig
echo NVIDIA_EGL_OK $V

# ManiSkill 3.0.1 pre-installed the way its docs say (installation page): pip install --upgrade mani_skill torch.
# Vulkan loader + ICD files as the page's templates (rendering only; state-only sim needs CUDA alone).
# Build with the nvidia runtime as the docker default (build.sh checks) so the warm-up step sees the GPU.
ARG BASE
FROM ${BASE}
ARG AGENT_HOME
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
USER root
RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libvulkan1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD": {\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.2.155"\n    }\n}\n' > /usr/share/vulkan/icd.d/nvidia_icd.json \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && mkdir -p /usr/local/lib/engine-venv && chown agent:agent /usr/local/lib/engine-venv
USER agent
RUN python -m venv /usr/local/lib/engine-venv
ENV VIRTUAL_ENV=/usr/local/lib/engine-venv \
    PATH=/usr/local/lib/engine-venv/bin:${AGENT_HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin \
    MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
RUN mkdir -p $VIRTUAL_ENV/.newenv && t0=$(date +%s) \
    && pip install --upgrade "mani_skill==3.0.1" torch \
    && echo $(( $(date +%s) - t0 )) > $VIRTUAL_ENV/.newenv/install_s \
    && pip freeze > $VIRTUAL_ENV/.newenv/freeze.txt
COPY --chown=agent:agent smoke.py /tmp/smoke.py
RUN t0=$(date +%s) && python /tmp/smoke.py && echo $(( $(date +%s) - t0 )) > $VIRTUAL_ENV/.newenv/warm_s && rm /tmp/smoke.py

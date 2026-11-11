# Genesis 1.4.6 pre-installed the way its docs say (installation page): PyTorch first, then pip install genesis-world.
# DEVIATION from the docs: genesis-world 1.4.6 imports matplotlib (genesis/utils/hybrid.py) but does not declare it, so
# `import genesis` fails after the documented install; `pip install matplotlib` is added (counted in install_s).
# The warm-up step fills the Quadrants kernel cache (rigid, one cloth, one SPH liquid step) on the GPU.
ARG BASE
FROM ${BASE}
ARG AGENT_HOME
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
USER root
RUN mkdir -p /usr/local/lib/engine-venv && chown agent:agent /usr/local/lib/engine-venv
USER agent
RUN python -m venv /usr/local/lib/engine-venv
ENV VIRTUAL_ENV=/usr/local/lib/engine-venv \
    PATH=/usr/local/lib/engine-venv/bin:${AGENT_HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin
RUN mkdir -p $VIRTUAL_ENV/.newenv && t0=$(date +%s) \
    && pip install torch \
    && pip install "genesis-world==1.4.6" \
    && pip install matplotlib \
    && echo $(( $(date +%s) - t0 )) > $VIRTUAL_ENV/.newenv/install_s \
    && pip freeze > $VIRTUAL_ENV/.newenv/freeze.txt
# DEVIATION: pymeshlab (needed to add a cloth / PBD / SPH mesh entity) dlopens libOpenGL.so.0, which the docs do not
# mention; without it `MeshSet.meshing_isotropic_explicit_remeshing` is missing and add_entity raises AttributeError.
USER root
RUN apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libopengl0 \
    && rm -rf /var/lib/apt/lists/*
USER agent
COPY --chown=agent:agent smoke.py /tmp/smoke.py
RUN t0=$(date +%s) && python /tmp/smoke.py && echo $(( $(date +%s) - t0 )) > $VIRTUAL_ENV/.newenv/warm_s && rm /tmp/smoke.py

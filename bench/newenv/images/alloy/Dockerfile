# Alloy pre-installed with its own install.sh from a staged release tree (the cold-agent kit's staging: make_release
# tree + wheelhouse of one commit), the way a stranger installs a clone: `bash install.sh --wheels DIR --extras envs`
# (envs = PyTorch + huggingface_hub, which install.sh adds by default on a GPU machine). The tree sits at ~/alloy and
# the venv at ~/alloy/.venv; run_one.sh later copies the same release tree over ~/alloy. The nvidia runtime is the
# docker default at build time, so install.sh's own quick start runs on the GPU and fills ~/.cache/warp; the
# smoke step warms the 16-world rigid scene too.
# Differs from the real cold runs: those get the cold-agent kit's staging of a named main commit, install nothing, and pip
# resolves alloy-engine from /opt/wheels (read-only mount); here the wheel is already installed from the same stage.
ARG BASE
FROM ${BASE}
ARG AGENT_HOME
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
COPY --chown=agent:agent release/ ${AGENT_HOME}/alloy/
COPY --chown=agent:agent wheels/ /tmp/wheels/
USER agent
RUN cd "$AGENT_HOME/alloy" && t0=$(date +%s) \
    && bash install.sh --wheels /tmp/wheels --extras envs \
    && mkdir -p .venv/.newenv && echo $(( $(date +%s) - t0 )) > .venv/.newenv/install_s \
    && .venv/bin/python -m pip freeze > .venv/.newenv/freeze.txt
USER root
RUN rm -rf /tmp/wheels
USER agent
ENV VIRTUAL_ENV=${AGENT_HOME}/alloy/.venv NEWENV_DIR=${AGENT_HOME}/alloy/.venv/.newenv \
    PATH=${AGENT_HOME}/alloy/.venv/bin:${AGENT_HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin
COPY smoke.py /tmp/smoke.py
RUN t0=$(date +%s) && python /tmp/smoke.py && echo $(( $(date +%s) - t0 )) > $NEWENV_DIR/warm_s
USER root
RUN rm /tmp/smoke.py
USER agent

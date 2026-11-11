# MuJoCo Playground 1.0.1 pre-installed the way its README says: uv venv, uv pip install "playground>=1.0"
# (pinned to ==1.0.1 with the rl extra, the README's training install). The MJX line (0.2.0) is a separate
# image, playground_mjx, if the extra arm runs.
ARG BASE
FROM ${BASE}
ARG AGENT_HOME
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
USER root
RUN mkdir -p /usr/local/lib/engine-venv && chown agent:agent /usr/local/lib/engine-venv
USER agent
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV VIRTUAL_ENV=/usr/local/lib/engine-venv \
    PATH=/usr/local/lib/engine-venv/bin:${AGENT_HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin
RUN t0=$(date +%s) \
    && uv venv --python 3.12 --seed $VIRTUAL_ENV \
    && mkdir -p $VIRTUAL_ENV/.newenv \
    && uv pip install --default-index https://pypi.org/simple "playground[rl]==1.0.1" \
    && echo $(( $(date +%s) - t0 )) > $VIRTUAL_ENV/.newenv/install_s \
    && uv pip freeze > $VIRTUAL_ENV/.newenv/freeze.txt
COPY --chown=agent:agent smoke.py /tmp/smoke.py
RUN t0=$(date +%s) && python /tmp/smoke.py && echo $(( $(date +%s) - t0 )) > $VIRTUAL_ENV/.newenv/warm_s && rm /tmp/smoke.py

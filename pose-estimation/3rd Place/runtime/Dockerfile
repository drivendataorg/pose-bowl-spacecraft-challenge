FROM mambaorg/micromamba:bookworm-slim

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

COPY apt.txt apt.txt
RUN apt-get update --fix-missing \
    && apt-get install -y apt-utils 2> /dev/null \
    && xargs -a apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /apt.txt

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yml
RUN micromamba install --name base --yes --file /tmp/env.yml && \
    micromamba clean --all --force-pkgs-dirs --yes

RUN mkdir -p /code_execution/submission
RUN chown -R ${MAMBA_USER}:${MAMBA_USER} /code_execution

COPY entrypoint.sh /entrypoint.sh

WORKDIR /code_execution
USER ${MAMBA_USER}
COPY tests /code_execution/tests

CMD ["bash", "/entrypoint.sh"]
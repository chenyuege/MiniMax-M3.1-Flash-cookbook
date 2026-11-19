ARG HCU_BASE_IMAGE
FROM ${HCU_BASE_IMAGE}

ARG SGLANG_REF=refs/tags/v0.5.23
ARG ADAPTER_SHA=ef63bbbd73a42812db9de1683e6720239c42fd68
ARG PLUGIN_SHA=0d777624d9cbafcb842c03cc76373fb495a282b8

WORKDIR /opt/minimax-m31
RUN git init sglang \
    && git -C sglang fetch --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_REF" \
    && git -C sglang checkout --detach FETCH_HEAD \
    && git init adapter-das \
    && git -C adapter-das fetch --depth 1 https://github.com/HYGON-AI/adapter-das.git "$ADAPTER_SHA" \
    && git -C adapter-das checkout --detach FETCH_HEAD \
    && git init sglang-plugin-das \
    && git -C sglang-plugin-das fetch --depth 1 https://github.com/HYGON-AI/sglang-plugin-das.git "$PLUGIN_SHA" \
    && git -C sglang-plugin-das checkout --detach FETCH_HEAD \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.23 \
       python3 -m pip install --no-deps --no-build-isolation -e ./sglang/python \
    && HCU_ADAPTER_HOST=none python3 -m pip install --no-deps --no-build-isolation \
       --target /opt/minimax-m31/plugin-site ./sglang-plugin-das \
    && python3 -m pip install --no-deps 'sglang-router==0.3.2'

ENV PYTHONPATH=/opt/minimax-m31/adapter-das:/opt/minimax-m31/plugin-site
COPY scripts/serve.sh /opt/minimax-m31/serve.sh

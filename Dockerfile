ARG HCU_BASE_IMAGE
FROM ${HCU_BASE_IMAGE}

ARG SGLANG_SHA=297e54fd105e974cdf401e5b4d20253a22c16f90
ARG ADAPTER_SHA=20d18cc6c550e32b119fe69a95db2c6ed83d7887
ARG PLUGIN_SHA=838ad34bfcb503a7f10f8cb60ab375505b079303

WORKDIR /opt/minimax-m31
RUN git init sglang \
    && git -C sglang fetch --depth 1 https://github.com/sgl-project/sglang.git "$SGLANG_SHA" \
    && git -C sglang checkout --detach FETCH_HEAD \
    && git init adapter-das \
    && git -C adapter-das fetch --depth 1 https://github.com/chenyuege/adapter-das.git "$ADAPTER_SHA" \
    && git -C adapter-das checkout --detach FETCH_HEAD \
    && git init sglang-plugin-das \
    && git -C sglang-plugin-das fetch --depth 1 https://github.com/chenyuege/sglang-plugin-das.git "$PLUGIN_SHA" \
    && git -C sglang-plugin-das checkout --detach FETCH_HEAD \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.23 \
       python3 -m pip install --no-deps --no-build-isolation -e ./sglang/python \
    && HCU_ADAPTER_HOST=none python3 -m pip install --no-deps --no-build-isolation \
       --target /opt/minimax-m31/plugin-site ./sglang-plugin-das \
    && python3 -m pip install --no-deps 'sglang-router==0.3.2'

ENV PYTHONPATH=/opt/minimax-m31/adapter-das:/opt/minimax-m31/plugin-site
COPY scripts/serve.sh /opt/minimax-m31/serve.sh

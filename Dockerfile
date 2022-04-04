# syntax=docker/dockerfile:1.4

FROM mambaorg/micromamba:0.22.0 as conda

WORKDIR /tmp
USER root

COPY conda-linux-64.lock ./

RUN --mount=type=cache,target=/opt/conda \
    micromamba create --always-copy --yes --file conda-linux-64.lock --prefix /env && \
    micromamba clean --all --yes

RUN \
  find -name '*.a' -delete && \
  rm -rf /env/conda-meta && \
  rm -rf /env/include && \
  rm /env/lib/libpython3.8.so.1.0 && \
  find / -name '__pycache__' -type d -exec rm -rf '{}' '+' && \
  rm -rf /env/lib/python3.8/site-packages/pip \
    /env/lib/python3.8/idlelib \
    /env/lib/python3.8/ensurepip \
    /env/lib/libasan.so.5.0.0 \
    /env/lib/libtsan.so.0.0.0 \
    /env/lib/liblsan.so.0.0.0 \
    /env/lib/libubsan.so.1.0.0 \
    /env/bin/x86_64-conda-linux-gnu-ld \
    /env/bin/sqlite3 \
    /env/bin/openssl \
    /env/share/terminfo && \
  find /env/lib/python3.8/site-packages/scipy -name 'tests' -type d -exec rm -rf '{}' '+' && \
  find /env/lib/python3.8/site-packages/numpy -name 'tests' -type d -exec rm -rf '{}' '+' && \
  find /env/lib/python3.8/site-packages/pandas -name 'tests' -type d -exec rm -rf '{}' '+' && \
  find /env/lib/python3.8/site-packages -name '*.pyx' -delete && \
  rm -rf /env/lib/python3.8/site-packages/uvloop/loop.c

FROM gcr.io/distroless/base-debian10

COPY --link --from=conda /env /env
COPY --link minder_utils ./minder_utils
CMD [ \
  "/env/bin/python" "/src/minder_utils/scripts/weekly_run.py" \
]
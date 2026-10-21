# Linux environment for decoding the Swabian Time Tagger .ttbin files.
# The Swabian Time Tagger Python API (needed for FileReader) has no macOS build,
# but PyPI has manylinux wheels for both x86_64 and aarch64, so this image builds
# natively on Apple Silicon (colima, arm64 VM) as well as on x86_64 Linux.
# Python 3.11 because scikit-learn 1.2.2 has no 3.12 wheels. scikit-learn is
# pinned to 1.2.2 on purpose: the GaussianMixture fit (random_state=42) of the
# published decode (Oct 2023) is reproduced exactly by 1.2.2 + numpy 1.26, while
# sklearn >= 1.3 converges to slightly different local optima.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba_cache

RUN pip install --no-cache-dir \
        "Swabian-TimeTagger==2.22.6" \
        "numpy==1.26.4" \
        "scipy==1.14.1" \
        "numba==0.60.0" \
        "scikit-learn==1.2.2" \
        "matplotlib==3.9.3" \
        "bokeh==3.6.2" \
        "orjson==3.10.12" \
        "pydantic==2.10.3" \
        "pyyaml==6.0.2" \
        "ipython==8.39.0" \
 && pip install --no-cache-dir --no-deps "snsphd==0.1.5"
# snsphd (only its plotting themes are used) declares numpy>=2, which
# scikit-learn 1.2.2 does not support; it works fine with numpy 1.26.

# The PyPI wheel installs the module as `Swabian.TimeTagger`; older Swabian
# installers (used when the data were taken, TT 2.8) exposed it as `TimeTagger`,
# which is what the decoder imports. Provide a thin alias module.
RUN printf 'from Swabian.TimeTagger import *  # noqa\n' \
        > /usr/local/lib/python3.11/site-packages/TimeTagger.py

# Mount the repository root at /work and pick the dataset folder with -w, e.g.
#   docker run --rm -v "$PWD":/work -w /work/ppm_20ghz/src_decode ppm-pnr-decode python redecode.py --db 48
WORKDIR /work

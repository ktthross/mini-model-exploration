#!/usr/bin/env bash
# Python 3.9 side environment for the optimal-tree and rule-list solvers, kept out of the main venv:
#   gosdt  — the index has prebuilt wheels only up to Python 3.12; the source build needs TBB and CMake
#   corels — does not build on 3.13. The vendored copy of the sdist drops one setup.py line
#            (`__builtins__.__NUMPY_SETUP__ = False`, fails under PEP 517 builds) and replaces the numpy
#            aliases removed in 1.24 (np.bool → bool) in corels/utils.py.
# The notebook never imports from here; it calls .venv/bin/python on a driver script and reads back CSV/text.
set -euo pipefail
cd "$(dirname "$0")"
export UV_INDEX_JFROG_PYPI_PASSWORD="${UV_INDEX_JFROG_PYPI_PASSWORD:-$JFROG_TOKEN}"
uv venv --python 3.9 .venv
uv pip install --python .venv/bin/python "numpy<2" "pandas<2.3" "scikit-learn<1.7" gosdt==1.0.4 setuptools wheel cython  # corels' pre-generated _corels.cpp predates numpy 2
GMP="$(brew --prefix gmp)"  # setup.py falls back to a slower no-GMP build if the header is not found
CPATH="$GMP/include" LIBRARY_PATH="$GMP/lib" uv pip install --python .venv/bin/python --no-build-isolation --no-deps ./corels-1.1.29
.venv/bin/python -W ignore -c "import corels, gosdt, sklearn, pandas; print('trees_and_rules env ok')"

#!/bin/bash
set -euo pipefail

echo "build.sh: starting"

# If apt-get is available (Debian/Ubuntu-like environment), install system deps.
# Many build environments (including some Vercel sandboxes) do NOT provide apt-get
# so we detect and skip this step gracefully.
if command -v apt-get >/dev/null 2>&1; then
	echo "build.sh: apt-get found — installing system packages (cmake, build-essential)"
	apt-get update
	apt-get install -y cmake build-essential || {
		echo "build.sh: apt-get install failed — continuing and hoping prebuilt wheels are available"
	}
else
	echo "build.sh: apt-get not found — skipping system package installation"
fi

echo "build.sh: upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "build.sh: installing Python requirements"
python -m pip install -r requirements.txt || {
	echo "build.sh: pip install failed — see output above. If dlib fails to build you'll need a prebuilt wheel or a different deployment approach."
	exit 1
}

echo "build.sh: verification (best-effort, non-fatal)"
python - <<'PY'
import sys
def try_import(name):
		try:
				m = __import__(name)
				print(f"{name} import OK, version=", getattr(m, '__version__', 'unknown'))
		except Exception as e:
				print(f"{name} import FAILED: {type(e).__name__}: {e}")

try_import('dlib')
try_import('face_recognition')
PY

echo "build.sh: finished"

# Install Python dependencies
pip install -r requirements.txt
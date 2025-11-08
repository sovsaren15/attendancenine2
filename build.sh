#!/bin/bash
set -euo pipefail

echo "build.sh: starting"

# If a service account JSON is provided via environment variable, write it to a temp file
# Use FIREBASE_SERVICE_ACCOUNT_JSON to pass the JSON content (base64 or raw). Do NOT commit keys to the repo.
if [ -n "${FIREBASE_SERVICE_ACCOUNT_JSON:-}" ]; then
	echo "build.sh: detected FIREBASE_SERVICE_ACCOUNT_JSON env var — writing credentials to /tmp/firebase_service_account.json"
	# write without showing the value in logs
	printf '%s' "$FIREBASE_SERVICE_ACCOUNT_JSON" > /tmp/firebase_service_account.json
	export GOOGLE_APPLICATION_CREDENTIALS=/tmp/firebase_service_account.json
fi

# If a path is provided instead, use it
if [ -n "${FIREBASE_SERVICE_ACCOUNT_PATH:-}" ]; then
	echo "build.sh: using FIREBASE_SERVICE_ACCOUNT_PATH"
	export GOOGLE_APPLICATION_CREDENTIALS="$FIREBASE_SERVICE_ACCOUNT_PATH"
fi

# If GOOGLE_APPLICATION_CREDENTIALS is already set in the environment, we don't overwrite it.
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
	echo "build.sh: GOOGLE_APPLICATION_CREDENTIALS is set"
fi

# If apt-get is available (Debian/Ubuntu-like environment), install system deps.
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
	echo "build.sh: pip install failed — see output above. If native packages fail to build you'll need a prebuilt wheel or a different deployment approach."
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

try_import('firebase_admin')
try_import('google.cloud.vision')
PY

echo "build.sh: finished"
#!/bin/bash
set -e

# Install system dependencies
apt-get update
apt-get install -y cmake build-essential

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Verify installations
python -c "import dlib; print('dlib version:', dlib.__version__)"
python -c "import face_recognition; print('face_recognition version:', face_recognition.__version__)"
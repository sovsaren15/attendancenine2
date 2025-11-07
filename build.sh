#!/bin/bash
# Install system dependencies for dlib
apt-get update
apt-get install -y cmake build-essential

# Install Python dependencies
pip install -r requirements.txt
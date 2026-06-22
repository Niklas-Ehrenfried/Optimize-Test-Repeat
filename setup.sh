#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

if [ "$1" != "cuda" ] && [ "$1" != "rocm" ]; then
    echo "Usage: ./setup.sh [cuda|rocm]"
    echo "Example: ./setup.sh rocm"
    exit 1
fi

TARGET="pyproject-$1.toml"

echo "Cleaning up old configurations..."
rm -f pyproject.toml
rm -f uv.lock
rm -rf .venv

echo "Linking $TARGET to pyproject.toml..."
ln -s "$TARGET" pyproject.toml

echo "Running uv sync for $1 environment..."
uv sync

echo "Environment setup complete. Activate with: source .venv/bin/activate"
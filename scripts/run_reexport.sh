#!/usr/bin/env bash
set -euo pipefail
# Build a container that contains TF 2.4.0 and runs the reexport script.

IMAGE_NAME=nxplorer-reexport:tf2.4

echo "Building re-export image ${IMAGE_NAME}..."
docker build -t ${IMAGE_NAME} -f Dockerfile.reexport .

echo "Running re-export (repo will be mounted into /workspace)..."
docker run --rm -v "$(pwd):/workspace" ${IMAGE_NAME}

echo "When the container finishes, check data/keras_model.resaved.h5 and data/keras_model.savedmodel/."

#!/bin/bash

# Ensure model.onnx exists
if [ ! -f "model.onnx" ]; then
    echo "Error: model.onnx not found in current directory!"
    exit 1
fi

echo "Building Docker image..."
docker build -t trt-converter ./trt_converter

echo "Debug: Checking for model.onnx inside container..."
docker run --rm -v "$(pwd):/workspace" trt-converter ls -lh /workspace/model.onnx

echo "Debug: Checking trtexec installation..."
docker run --rm -v "$(pwd):/workspace" trt-converter trtexec --help | head -n 5

docker run --rm -v "$(pwd):/workspace" trt-converter \
    trtexec --onnx=/workspace/model.onnx \
            --saveEngine=/workspace/model.trt \
            --verbose

if [ -f "model.trt" ]; then
    echo "Success! model.trt created."
else
    echo "Conversion failed."
    exit 1
fi

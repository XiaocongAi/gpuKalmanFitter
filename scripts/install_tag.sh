#!/bin/sh

REPO_LINK=https://github.com/XiaocongAi/gpuKalmanFitter

if [ "$#" -ne 1 ]; then
    echo "Missing tag version. (e.g. 0.1)"
    exit 1
fi
echo "Install gpuKalmanFitter tag $1 ..."

mkdir app
cd app
echo "Downloading tag $1 ..."
wget $REPO_LINK/archive/v$1.tar.gz
echo "Download completed!"

echo "Unpacking the archive..."
tar -xzf v$1.tar.gz
echo "Unpacked!"

echo "Setup and build the project..."
cd gpuKalmanFitter-$1 && mkdir build && cd build
cmake ..
echo "Build complete!"  

echo "Installing to $(pwd)"
make install
echo "Executables successfully installed!"


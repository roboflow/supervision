#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

# Download the market-square.mp4 file from Google Drive
gdown -O "$DIR/data/market-square.mp4" "https://drive.google.com/uc?id=1vVrEVMxucHgqGd7vAa501ASojbeGPhIr"

#!/bin/bash

echo "Setting up environment for Grounded SAM2 Image Segmentation..."

# Create necessary directories
echo "Creating directories..."
mkdir -p images
mkdir -p ../checkpoints/

# Download sample images
echo "Downloading sample images..."
wget -q -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg
wget -q -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg

# Check if downloads were successful
if [ -f "images/truck.jpg" ] && [ -f "images/groceries.jpg" ]; then
    echo "✓ Sample images downloaded successfully"
else
    echo "✗ Failed to download sample images"
    exit 1
fi

# Download SAM 2 model
echo "Downloading SAM 2 model..."
wget -q -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

if [ -f "../checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "✓ SAM 2 model downloaded successfully"
else
    echo "✗ Failed to download SAM 2 model"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo "You can now run: python inference_example.py"

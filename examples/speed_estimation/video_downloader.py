import os

from supervision.assets import VideoAssets, download_assets

if not os.path.exists("data"):
    os.makedirs("data")
os.chdir("data")
download_assets(VideoAssets.VEHICLES)

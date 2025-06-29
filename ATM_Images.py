"""
Script to Download Clean ATM Front View Images

This script uses the `bing_image_downloader` package to fetch clean,
front-facing images of ATMs and organize them into a target directory
for later use in training or visual comparison.

Steps:
1. Download ~50 images from Bing using the search query "ATM front view".
2. Move those images into `data/clean_atms/` directory.
3. Clean up any temporary folders created during download.
"""

from bing_image_downloader import downloader  # Image search & download from Bing
import os                                     # File system operations
import shutil                                 # High-level file/folder manipulation

# === Step 1: Ensure output directory exists ===
os.makedirs("data/clean_atms", exist_ok=True)

# === Step 2: Download images from Bing ===
downloader.download(
    "ATM front view",     # Search query to find front-view ATM images
    limit=50,             # Number of images to download
    output_dir='data',    # Where to download (temporary root folder)
    adult_filter_off=True,# Disable adult filter (ATM is safe anyway)
    force_replace=False,  # Don't redownload if already exists
    timeout=60,           # Network timeout for downloads
    verbose=True          # Print progress to console
)

# === Step 3: Move downloaded files to final destination ===
download_path = "data/ATM front view"  # Bing saves images here by query name
if os.path.exists(download_path):
    for filename in os.listdir(download_path):
        shutil.move(os.path.join(download_path, filename), "data/clean_atms")
    shutil.rmtree(download_path)  # Remove the now-empty query folder

print("✅ Clean ATM images downloaded to: data/clean_atms/")
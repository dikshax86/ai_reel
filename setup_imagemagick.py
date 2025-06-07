import os
import sys
import requests
import zipfile
from pathlib import Path
import subprocess
from moviepy.config import change_settings

def download_imagemagick():
    """Download ImageMagick portable version"""
    # Use portable version instead
    urls = [
        "https://imagemagick.org/archive/binaries/ImageMagick-7.1.1-Q16-HDRI-x64-static.zip",
        "https://mirror.imagemagick.org/archive/binaries/ImageMagick-7.1.1-Q16-HDRI-x64-static.zip"
    ]
    
    print("Downloading ImageMagick portable version...")
    zip_path = "imagemagick.zip"
    
    for url in urls:
        try:
            print(f"Trying {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percentage = int((downloaded / total_size) * 100)
                        print(f"\rDownloading: {percentage}%", end='')
            print("\nDownload complete!")
            return zip_path
            
        except Exception as e:
            print(f"Failed to download from {url}: {str(e)}")
            continue
            
    raise Exception("Failed to download ImageMagick from all mirrors")

def setup_imagemagick():
    """Setup ImageMagick in the local directory"""
    try:
        # Create imagemagick directory if it doesn't exist
        imagemagick_dir = Path("imagemagick")
        imagemagick_dir.mkdir(exist_ok=True)
        
        # Download portable version
        zip_path = download_imagemagick()
        
        # Extract to imagemagick directory
        print("Extracting ImageMagick...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(imagemagick_dir)
        
        # Clean up
        os.remove(zip_path)
        
        # Find magick.exe in the extracted files
        magick_exe = None
        for root, dirs, files in os.walk(imagemagick_dir):
            if "magick.exe" in files:
                magick_exe = os.path.join(root, "magick.exe")
                break
        
        if magick_exe:
            # Update moviepy settings
            change_settings({"IMAGEMAGICK_BINARY": os.path.abspath(magick_exe)})
            print(f"ImageMagick setup complete! Path: {magick_exe}")
            return magick_exe
        else:
            raise Exception("Could not find magick.exe in the extracted files")
            
    except Exception as e:
        print(f"Error setting up ImageMagick: {str(e)}")
        raise

if __name__ == "__main__":
    setup_imagemagick()

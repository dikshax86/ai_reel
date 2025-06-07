import os
import requests
import zipfile
import winreg
import sys
import subprocess
from pathlib import Path

def download_ffmpeg():
    """Download FFmpeg from GitHub releases"""
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    print("Downloading FFmpeg...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    zip_path = "ffmpeg.zip"
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return zip_path

def extract_ffmpeg(zip_path):
    """Extract FFmpeg zip file"""
    print("Extracting FFmpeg...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("ffmpeg")
    
    # Find the bin directory
    ffmpeg_dir = next(Path("ffmpeg").glob("ffmpeg-*"))
    return ffmpeg_dir / "bin"

def add_to_path(bin_path):
    """Add FFmpeg to system PATH"""
    print("Adding FFmpeg to PATH...")
    
    # Get the current PATH
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
        path = winreg.QueryValueEx(key, "Path")[0]
        
        # Add FFmpeg to PATH if not already present
        bin_path_str = str(bin_path.absolute())
        if bin_path_str not in path:
            new_path = f"{path};{bin_path_str}"
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            
        winreg.CloseKey(key)
        
        # Notify Windows of the change
        subprocess.run(['setx', 'Path', new_path], check=True)
        
    except Exception as e:
        print(f"Error updating PATH: {e}")
        print("\nPlease manually add the following path to your system PATH:")
        print(bin_path.absolute())
        return False
    
    return True

def cleanup(zip_path):
    """Clean up downloaded zip file"""
    try:
        os.remove(zip_path)
    except:
        pass

def main():
    try:
        # Download FFmpeg
        zip_path = download_ffmpeg()
        
        # Extract it
        bin_path = extract_ffmpeg(zip_path)
        
        # Add to PATH
        success = add_to_path(bin_path)
        
        # Cleanup
        cleanup(zip_path)
        
        if success:
            print("\nFFmpeg has been successfully installed!")
            print("Please restart your terminal/IDE for the PATH changes to take effect.")
        else:
            print("\nFFmpeg files have been extracted but couldn't be added to PATH automatically.")
            print(f"Please manually add this path to your system PATH: {bin_path.absolute()}")
        
    except Exception as e:
        print(f"Error during FFmpeg installation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

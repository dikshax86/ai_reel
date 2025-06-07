import sys
print(f"Using Python: {sys.executable}")
try:
    from moviepy import VideoFileClip
    print("SUCCESS: moviepy.editor imported successfully!")
except ImportError as e:
    print(f"ERROR: Failed to import moviepy.editor: {e}")
except Exception as e:
    print(f"UNEXPECTED ERROR: {e}")

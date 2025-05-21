import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
try:
    import whisper
    print(f"Whisper imported successfully from {whisper.__file__}")
except Exception as e:
    print(f"Error importing whisper: {e}")
    import traceback
    traceback.print_exc()
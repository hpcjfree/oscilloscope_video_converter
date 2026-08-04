@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name OscilloscopeVideoConverter ^
  --collect-all cv2 ^
  oscilloscope_video_converter.py
pause

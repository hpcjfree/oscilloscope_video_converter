@echo off
cd /d "%~dp0"
python oscilloscope_video_converter.py
if errorlevel 1 pause

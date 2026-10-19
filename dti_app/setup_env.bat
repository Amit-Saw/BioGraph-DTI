@echo off
rem Creates .venv and installs everything.   setup_env.bat [cpu|cu121|cu118]
cd /d "%~dp0"
set BUILD=%1
if "%BUILD%"=="" set BUILD=cpu
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/%BUILD%
pip install -r requirements.txt
echo Done. Next: python scripts\setup_data.py --src "your Drive folder"

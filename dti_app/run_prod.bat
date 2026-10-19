@echo off
rem Production server on Windows (waitress). Change the port if needed.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
waitress-serve --listen=0.0.0.0:8000 --threads=8 wsgi:app

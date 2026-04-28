@echo off
cd /d "%~dp0"
python -m uvicorn backend.app:app --reload

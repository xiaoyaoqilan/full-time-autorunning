@echo off
cd /d "%~dp0"
set "FLYWHEEL_PY=%~dp0..\AIInfra_env\Scripts\python.exe"
if not exist "%FLYWHEEL_PY%" (
  echo [run_daily] Python environment missing. No generation started.
  exit /b 1
)
if not exist ".env" (
  echo [run_daily] Missing .env. No generation started.
  exit /b 1
)
if not exist "novel\bible.md" (
  echo [run_daily] Missing novel\bible.md. No generation started.
  exit /b 1
)
"%FLYWHEEL_PY%" -c "import sys,pipeline; e=pipeline.load_env(); sys.exit(0 if e.get('LLM_BASE_URL') and e.get('LLM_API_KEY') else 1)"
if errorlevel 1 (
  echo [run_daily] LLM configuration incomplete. No generation started.
  exit /b 1
)
"%FLYWHEEL_PY%" novel.py --gen 1
if errorlevel 1 exit /b 1
"%FLYWHEEL_PY%" pipeline.py --weread-ideas 3
if errorlevel 1 exit /b 1
if "%GEN_TECH_ARTICLE%"=="0" goto skip_gen
"%FLYWHEEL_PY%" pipeline.py --gen
if errorlevel 1 exit /b 1
:skip_gen
echo [run_daily] Commands finished. Check outputs. Publication is manual.

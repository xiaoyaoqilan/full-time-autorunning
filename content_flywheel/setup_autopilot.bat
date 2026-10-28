@echo off
rem content_flywheel autopilot scheduler (ASCII only)
rem Creates Windows scheduled tasks: generate twice daily + nightly feedback
set PY=C:\Users\Lukas\WorkBuddy\2026-10-15-09-43-08\AIInfra_env\Scripts\python.exe
set PIPE=C:\Users\Lukas\WorkBuddy\2026-10-15-09-43-08\content_flywheel\pipeline.py

schtasks /Create /F /SC DAILY /ST 09:00 /TN ContentFlywheel_AM /TR "\"%PY%\" \"%PIPE%\" --autopilot"
schtasks /Create /F /SC DAILY /ST 21:00 /TN ContentFlywheel_PM /TR "\"%PY%\" \"%PIPE%\" --autopilot"
schtasks /Create /F /SC DAILY /ST 22:00 /TN ContentFlywheel_FB /TR "\"%PY%\" \"%PIPE%\" --feedback"

echo.
echo Scheduled tasks created: ContentFlywheel_AM / PM / FB
echo Check:  schtasks /Query /TN ContentFlywheel_AM
echo Remove: schtasks /Delete /TN ContentFlywheel_AM /F
pause

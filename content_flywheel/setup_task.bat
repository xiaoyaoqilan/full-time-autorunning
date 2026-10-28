@echo off
REM setup_task.bat · 注册 Windows 计划任务: 每天 08:30 自动跑 content_flywheel 日更
REM 只生成不入库不发布, 守"说发才发"红线。卸载: schtasks /Delete /TN ContentFlywheelDaily /F
schtasks /Create /TN "ContentFlywheelDaily" /TR "\"%~dp0run_daily.bat\"" /SC DAILY /ST 08:30 /F
if %errorlevel%==0 (
  echo [setup] 计划任务 ContentFlywheelDaily 已注册: 每天 08:30 自动生成
  schtasks /Query /TN "ContentFlywheelDaily"
) else (
  echo [setup] 注册失败, 试试右键"以管理员身份运行"本脚本
)
pause

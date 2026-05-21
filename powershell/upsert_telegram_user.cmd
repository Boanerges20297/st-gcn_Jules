@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0powershell\upsert_telegram_user.ps1" %*

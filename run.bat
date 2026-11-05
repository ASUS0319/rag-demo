@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 优先使用项目内的虚拟环境，避免用到系统 Python（依赖没装）
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

rem 把 Streamlit 的配置目录指到项目内，避开 C:\Users\<用户名>\.streamlit 的写入权限问题
set "USERPROFILE=%~dp0"
set "USERPROFILE=%USERPROFILE:~0,-1%"
set "HOME=%USERPROFILE%"
set "PYTHONUTF8=1"

echo ============================================
echo  解释器：%PY%
echo  配置目录：%USERPROFILE%\.streamlit
echo  启动中，浏览器会自动打开 http://localhost:8501
echo  停止服务：按 Ctrl+C
echo ============================================
echo.

"%PY%" -u -m streamlit run app.py

echo.
echo 服务已退出。按任意键关闭窗口。
pause >nul

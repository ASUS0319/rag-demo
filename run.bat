@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 把 Streamlit 的配置目录指到项目内，避开 C:\Users\<用户名>\.streamlit 的写入权限问题
set "USERPROFILE=%~dp0"
set "USERPROFILE=%USERPROFILE:~0,-1%"
set "HOME=%USERPROFILE%"
set "PYTHONUTF8=1"

echo ============================================
echo  配置目录：%USERPROFILE%\.streamlit
echo  启动中，浏览器会自动打开 http://localhost:8501
echo  停止服务：按 Ctrl+C
echo ============================================
echo.

python -u -m streamlit run app.py

echo.
echo 服务已退出。按任意键关闭窗口。
pause >nul

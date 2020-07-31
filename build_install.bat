:: clearing dist folder
cd /d dist
for /F "delims=" %%i in ('dir /b') do (rmdir "%%i" /s/q || del "%%i" /s/q)
cd ..

:: building .whl package
python setup.py sdist bdist_wheel

:: installing package
pip install --upgrade dist/nn_sim-0.0.1-py3-none-any.whl

pause

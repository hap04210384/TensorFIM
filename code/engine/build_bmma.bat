@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -arch=native -std=c++17 -Xcompiler /EHsc,/openmp -o run\CoParaCG_bmma.exe src\kernel_bmma.cu

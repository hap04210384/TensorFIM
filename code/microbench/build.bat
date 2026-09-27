@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -arch=sm_86 -std=c++17 -o bmma_bench.exe bmma_bench.cu

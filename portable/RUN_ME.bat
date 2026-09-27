@echo off
REM ===========================================================================
REM  TensorFIM USB one-click cross-GPU validation  (v2: prebuilt-first)
REM  Usage:  double-click RUN_ME.bat          (full run, 6 configs, ~1-1.5 h)
REM          RUN_ME.bat quick                (3 configs, ~30 min)
REM          RUN_ME.bat check                (environment check only, ~1 min)
REM
REM  v2: if prebuilt\ binaries are present, NO CUDA/VS installation is needed
REM      at all (static-linked fat binaries for sm_80/86/89/90/100/120 + PTX).
REM      Source build with winget auto-install remains as fallback.
REM  Everything else is automatic. Results -> runs\<machine>\<timestamp>\.
REM ===========================================================================
chcp 65001 >nul
title TensorFIM USB validation
setlocal EnableDelayedExpansion

set "PKG=%~dp0"
REM PKG ends with a trailing backslash, which breaks quoted PowerShell args
REM (\" is parsed as an escaped quote). Keep a backslash-free copy for -Root.
set "PKGR=%PKG:~0,-1%"
set "APP=%PKG%anyfim_bmma"
set "LOGDIR=%PKG%logs"
set "REPS=3"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if "%COMPUTERNAME%"=="" set "COMPUTERNAME=UNKNOWNPC"

set "CONFIGS=%PKG%scripts\configs.txt"
if /i "%~1"=="quick" set "CONFIGS=%PKG%scripts\configs_quick.txt"
REM hidden feature: RUN_ME.bat <path-to-a-config-file> overrides the config list
if exist "%~1" set "CONFIGS=%~f1"

REM -------- prebuilt binaries available? then no toolchain is needed --------
set "PREBUILT=0"
if exist "%PKG%prebuilt\CoParaCG_bmma.exe" if exist "%PKG%prebuilt\CoParaCG_baseline.exe" if exist "%PKG%prebuilt\smcheck.exe" set "PREBUILT=1"

echo ============================================================
echo  TensorFIM cross-GPU validation (USB edition, v2)
if "%PREBUILT%"=="1" (echo  Mode: PREBUILT - no CUDA/VS needed on this machine) else (echo  Mode: SOURCE BUILD - will need CUDA toolkit + VS C++)
echo  Reps per engine: %REPS%
echo ============================================================
echo.

if "%PREBUILT%"=="1" goto :probe_prebuilt

REM ---------------- Step 1: locate CUDA (auto-install if missing) ----------
:check_cuda
echo [1/5] Checking CUDA toolkit...
set "NVCC="
for %%V in (v13.0 v12.9 v12.8 v12.7 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
    if not defined NVCC if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin\nvcc.exe" (
        set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin\nvcc.exe"
    )
)
if not defined NVCC (
    if "%TRIED_CUDA%"=="1" goto :cuda_manual
    echo       CUDA not found. Trying automatic installation via winget
    echo       ^(about 3 GB download, 10-20 min; click "Yes" on UAC prompts^)...
    where winget >nul 2>&1
    if errorlevel 1 goto :cuda_manual
    winget install --id Nvidia.CUDA -v 12.8 -e --accept-package-agreements --accept-source-agreements --override "-s"
    set "TRIED_CUDA=1"
    goto :check_cuda
)
goto :cuda_ok
:cuda_manual
echo.
echo [ERROR] CUDA toolkit not found and could not be auto-installed.
echo         Please install CUDA 12.8 or newer manually from:
echo         https://developer.nvidia.com/cuda-downloads
echo         ^(RTX 50-series REQUIRES ^>= 12.8^)
echo         Then run this script again.
pause
exit /b 1
:cuda_ok
echo       Found: %NVCC%

REM ---------------- Step 2: locate Visual Studio (auto-install if missing) --
:check_vs
echo [2/5] Checking Visual Studio C++ environment...
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
REM VS 2019 (v142 toolset) is still supported by CUDA 12.x - accept it as-is
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    if "%TRIED_VS%"=="1" goto :vs_manual
    echo       VS 2022 C++ environment not found. Trying automatic installation
    echo       via winget ^(about 7 GB download, 20-40 min; click "Yes" on UAC^)...
    where winget >nul 2>&1
    if errorlevel 1 goto :vs_manual
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--quiet --wait --add Microsoft.VisualStudio.Workload.NativeDesktop;includeRecommended"
    set "TRIED_VS=1"
    goto :check_vs
)
goto :vs_ok
:vs_manual
echo.
echo [ERROR] Visual Studio 2022 with C++ workload not found.
echo         Install VS 2022 Community ^(free^) with "Desktop development with C++":
echo         https://visualstudio.microsoft.com/vs/community/
echo         Then run this script again.
pause
exit /b 1
:vs_ok
call "%VCVARS%" >nul 2>&1
echo       OK.

REM ---------------- Step 3a: probe GPU (source-build path: compile probe) --
echo [3/5] Probing GPU...
"%NVCC%" -O2 -arch=native -o "%LOGDIR%\smcheck.exe" "%PKG%scripts\smcheck.cu" >"%LOGDIR%\smcheck_build.log" 2>&1
if errorlevel 1 (
    REM Installed CUDA may be too old for this GPU (e.g. CUDA 12.5 + RTX 5090).
    REM Detect the signature error and auto-install a newer toolkit alongside.
    findstr /i /C:"unsupported gpu architecture" "%LOGDIR%\smcheck_build.log" >nul
    if not errorlevel 1 (
        if not "%TRIED_CUDA%"=="1" (
            echo       This GPU is newer than the installed CUDA toolkit.
            echo       Auto-installing CUDA 12.8 alongside ^(old version untouched^)...
            where winget >nul 2>&1
            if not errorlevel 1 (
                winget install --id Nvidia.CUDA -v 12.8 -e --accept-package-agreements --accept-source-agreements --override "-s"
            )
            set "TRIED_CUDA=1"
            goto :check_cuda
        )
    )
    echo [ERROR] Failed to build the GPU probe. See logs\smcheck_build.log
    pause
    exit /b 1
)
"%LOGDIR%\smcheck.exe" > "%LOGDIR%\sm.txt" 2>&1
goto :probe_done

:probe_prebuilt
echo [1/5] Toolchain check SKIPPED (prebuilt binaries, static-linked).
echo [2/5] GPU probing with prebuilt probe...
"%PKG%prebuilt\smcheck.exe" > "%LOGDIR%\sm.txt" 2>&1
if errorlevel 1 (
    echo [ERROR] Prebuilt probe failed - is an NVIDIA driver installed?
    pause
    exit /b 1
)

:probe_done
set "SMMAJOR=0"
set "DEVNAME=UnknownGPU"
for /f "tokens=2 delims==" %%a in ('findstr /C:"SMMAJOR=" "%LOGDIR%\sm.txt"') do set "SMMAJOR=%%a"
for /f "tokens=2 delims==" %%a in ('findstr /C:"DEVNAME=" "%LOGDIR%\sm.txt"') do set "DEVNAME=%%a"
echo [3/5] GPU: %DEVNAME%
if "%SMMAJOR%"=="0" (
    echo [ERROR] No CUDA-capable NVIDIA GPU detected on this machine.
    pause
    exit /b 1
)
set "BMMA=1"
if %SMMAJOR% LSS 8 (
    set "BMMA=0"
    echo       NOTE: compute capability major=%SMMAJOR% ^< 8: tensor-core engine
    echo             unsupported on this GPU; running BASELINE engine only.
) else (
    echo       Both engines will be tested.
)

REM -------- check-only mode stops here --------
if /i "%~1"=="check" (
    echo.
    echo Environment check PASSED. Run RUN_ME.bat ^(or RUN_ME.bat quick^) to start.
    pause
    exit /b 0
)

REM -------- per-machine run folder: never overwrite another machine's data --
for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%a"
set "GPU_TAG=%DEVNAME:NVIDIA =%"
set "GPU_TAG=%GPU_TAG: =_%"
set "RUNDIR=%PKG%runs\%COMPUTERNAME%_%GPU_TAG%_%TS%"
mkdir "%RUNDIR%"
copy /y "%LOGDIR%\sm.txt" "%RUNDIR%\" >nul
echo.
echo       All results go to: runs\%COMPUTERNAME%_%GPU_TAG%_%TS%\
echo       ^(each machine gets its own folder; nothing is overwritten^)

REM -------- full GPU snapshot: driver, clocks, power limit, BIOS ------------
REM -------- (platform-table footnote material for the paper) ----------------
nvidia-smi -q > "%RUNDIR%\nvidia_smi_full.txt" 2>&1
REM -------- host snapshot: OS build + total RAM (CPU is in every engine log) -
powershell -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,BuildNumber,OSArchitecture | Format-List; Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory | Format-List; Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List" > "%RUNDIR%\host_info.txt" 2>&1

REM -------- stage executables into the run directory ------------------------
if "%PREBUILT%"=="1" (
    copy /y "%PKG%prebuilt\CoParaCG_baseline.exe" "%APP%\run\" >nul
    copy /y "%PKG%prebuilt\vcomp140.dll" "%APP%\run\" >nul
    if "%BMMA%"=="1" copy /y "%PKG%prebuilt\CoParaCG_bmma.exe" "%APP%\run\" >nul
)

REM ---------------- Step 4: build (if needed) + run loop --------------------
echo [4/5] Running configs. Do NOT use this computer until finished
echo       ^(timing data becomes useless if the machine is busy^).
echo.
for /f "usebackq tokens=1,2" %%a in ("%CONFIGS%") do call :run_config %%a %%b
goto :sustain

:run_config
set "DS=%~1"
set "SUP=%~2"
if "%DS%"=="" exit /b 0
if "%DS:~0,1%"=="#" exit /b 0
echo ------------------------------------------------------------
echo   dataset: %DS%   minsup: %SUP%
powershell -NoProfile -ExecutionPolicy Bypass -File "%PKG%scripts\set_dataset.ps1" -Ds "%DS%" -Sup "%SUP%" -Root "%PKGR%" >nul
if "%PREBUILT%"=="0" (
    echo   building baseline...
    pushd "%APP%"
    "%NVCC%" -O3 -arch=native -std=c++17 -Xcompiler /EHsc,/openmp -o run\CoParaCG_baseline.exe src\kernel_bitmap.cu >"%RUNDIR%\build_%DS%_baseline.log" 2>&1
    if errorlevel 1 (
        echo   [ERROR] baseline build failed, see run folder log
        popd
        exit /b 1
    )
    popd
)
set "BMMA_OK=0"
if "%BMMA%"=="1" (
    if "%PREBUILT%"=="1" (
        set "BMMA_OK=1"
    ) else (
        echo   building bmma...
        pushd "%APP%"
        "%NVCC%" -O3 -arch=native -std=c++17 -Xcompiler /EHsc,/openmp -o run\CoParaCG_bmma.exe src\kernel_bmma.cu >"%RUNDIR%\build_%DS%_bmma.log" 2>&1
        if errorlevel 1 (
            echo   [ERROR] bmma build failed, see run folder log - skipping bmma for this dataset
            popd
        ) else (
            popd
            set "BMMA_OK=1"
        )
    )
)
for /l %%r in (1,1,%REPS%) do (
    echo   run baseline rep %%r/%REPS% ...
    pushd "%APP%\run"
    CoParaCG_baseline.exe "..\TransactionSets\%DS%" "%SUP%" >"%RUNDIR%\%DS%_%SUP%_baseline_r%%r.log" 2>&1
    popd
    if "%BMMA_OK%"=="1" (
        echo   run bmma     rep %%r/%REPS% ...
        pushd "%APP%\run"
        CoParaCG_bmma.exe "..\TransactionSets\%DS%" "%SUP%" >"%RUNDIR%\%DS%_%SUP%_bmma_r%%r.log" 2>&1
        popd
    )
)
exit /b 0

REM ---------------- Step 4b: sustained-load throttle probe ------------------
:sustain
if not "%BMMA%"=="1" goto :report
set "BENCH=%PKG%prebuilt\bmma_bench.exe"
if not exist "%BENCH%" set "BENCH=%LOGDIR%\bmma_bench.exe"
if not exist "%BENCH%" if "%PREBUILT%"=="0" (
    echo   building microbench for sustain probe...
    "%NVCC%" -O3 -arch=native -std=c++17 -Xcompiler /EHsc -o "%LOGDIR%\bmma_bench.exe" "%PKG%scripts\bmma_bench.cu" >"%LOGDIR%\bench_build.log" 2>&1
)
if not exist "%BENCH%" (
    echo   [WARN] no microbench available, sustain probe skipped
    goto :report
)
echo.
echo   Sustained BMMA probe (30 s, detects export-limiter throttling)...
"%BENCH%" sustain 1048576 16384 1024 30 > "%RUNDIR%\sustain_probe.csv" 2>&1
findstr /C:"verdict" "%RUNDIR%\sustain_probe.csv"

echo.
echo   Peak BMMA throughput sweep (8 shapes, a few minutes)...
echo   ^(microarchitecture-level tensor-core numbers for this GPU^)
"%BENCH%" > "%RUNDIR%\microbench_sweep.log" 2>&1
if errorlevel 1 (echo   [WARN] sweep failed, see microbench_sweep.log) else (echo   sweep done.)

:report
REM restore default dataset setting
powershell -NoProfile -ExecutionPolicy Bypass -File "%PKG%scripts\set_dataset.ps1" -Ds "pumsb_x256.txt" -Sup "0.8" -Root "%PKGR%" >nul

echo.
echo [5/5] Verifying results against archived references, writing REPORT.txt ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%PKG%scripts\compare.ps1" -ConfigFile "%CONFIGS%" -LogDir "%RUNDIR%"

echo.
echo ============================================================
echo  DONE. Results are in:  runs\%COMPUTERNAME%_%GPU_TAG%_%TS%\
echo  Bring the whole USB drive back - every machine's data is
echo  kept in its own folder under runs\.
echo  It is now safe to remove the USB drive.
echo ============================================================
pause

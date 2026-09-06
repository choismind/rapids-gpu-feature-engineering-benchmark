<#
.SYNOPSIS
  이 폴더의 스크립트를 WSL2 의 RAPIDS venv 로 실행한다.

.DESCRIPTION
  RAPIDS(cudf/cuml)는 manylinux 휠만 존재해서 Windows 네이티브 파이썬으로는 못 돌린다.
  이 스크립트가 WSL2 안의 venv 로 넘겨준다.

  작업 폴더는 하드코딩하지 않고 이 스크립트가 놓인 위치를 wslpath 로 변환해 쓴다.
  배포판·인터프리터 경로는 환경변수로 바꿀 수 있다.

    RAPIDS_WSL_DISTRO   기본 Ubuntu-24.04
    RAPIDS_WSL_PYTHON   기본 $HOME/rapids-venv/bin/python

.EXAMPLE
  .\run.ps1 00_env.py                          # CPU 백엔드
  .\run.ps1 01_cudf_features.py -Gpu           # python -m cudf.pandas 로 실행
  .\run.ps1 04_cuml_accel_sklearn.py -Accel    # python -m cuml.accel 로 실행
  .\run.ps1 05_benchmark_features.py -Gpu -Rows 10000000
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,

    # pandas 를 cudf.pandas 로 대체해서 실행
    [switch]$Gpu,

    # scikit-learn 을 cuml.accel 로 가속해서 실행
    [switch]$Accel,

    # 벤치마크 행 수 (N_ROWS 환경변수로 전달)
    [int]$Rows = 0
)

$distro = if ($env:RAPIDS_WSL_DISTRO) { $env:RAPIDS_WSL_DISTRO } else { 'Ubuntu-24.04' }
$venvPy = if ($env:RAPIDS_WSL_PYTHON) { $env:RAPIDS_WSL_PYTHON } else { '$HOME/rapids-venv/bin/python' }

# 이 스크립트가 놓인 폴더를 WSL 경로로 변환한다 (경로를 하드코딩하지 않는다).
# wsl.exe 에 백슬래시 경로를 인자로 넘기면 이스케이프로 먹히므로 직접 변환한다.
if ($PSScriptRoot -notmatch '^([A-Za-z]):\\') {
    Write-Error "드라이브 문자로 시작하는 로컬 경로에서 실행하세요 (현재: $PSScriptRoot)."
    exit 1
}
$drive = $Matches[1].ToLower()
$workdir = "/mnt/$drive" + ($PSScriptRoot.Substring(2) -replace '\\', '/')

$prefix = ''
if ($Gpu)   { $prefix = '-m cudf.pandas' }
if ($Accel) { $prefix = '-m cuml.accel' }

$envPart = ''
if ($Rows -gt 0) { $envPart = "N_ROWS=$Rows " }

$cmd = "cd '$workdir' && ${envPart}$venvPy $prefix $Script"
Write-Host "wsl -d $distro -- bash -lc `"$cmd`"" -ForegroundColor DarkGray
wsl -d $distro -- bash -lc $cmd

<# 이 폴더의 검증 스크립트를 순서대로 실행한다 (벤치마크 제외). #>
$ErrorActionPreference = 'Continue'

$steps = @(
    @{ s = '00_env.py';                    gpu = $false; accel = $false; note = 'CPU 기준 환경 확인' },
    @{ s = '00_env.py';                    gpu = $true;  accel = $false; note = 'GPU 백엔드 확인' },
    @{ s = '01_cudf_features.py';          gpu = $false; accel = $false; note = '① cuDF 피처 — CPU' },
    @{ s = '01_cudf_features.py';          gpu = $true;  accel = $false; note = '① cuDF 피처 — GPU' },
    @{ s = '02_target_encoding_manual.py'; gpu = $true;  accel = $false; note = '② 타깃 인코딩 함정 재현' },
    @{ s = '03_cuml_target_encoder.py';    gpu = $false; accel = $false; note = '③ cuML TargetEncoder (네이티브)' },
    @{ s = '04_cuml_accel_sklearn.py';     gpu = $false; accel = $false; note = '④ sklearn — 가속 없음' },
    @{ s = '04_cuml_accel_sklearn.py';     gpu = $false; accel = $true;  note = '④ sklearn — cuml.accel (③④절을 위 실행과 대조할 것)' }
)

foreach ($step in $steps) {
    Write-Host ''
    Write-Host ('#' * 78) -ForegroundColor Cyan
    Write-Host "# $($step.note)  ->  $($step.s)" -ForegroundColor Cyan
    Write-Host ('#' * 78) -ForegroundColor Cyan
    & "$PSScriptRoot\run.ps1" -Script $step.s -Gpu:$step.gpu -Accel:$step.accel
}

Write-Host ''
Write-Host '벤치마크는 시간이 걸리므로 따로 실행하세요:' -ForegroundColor Yellow
Write-Host '  .\run.ps1 05_benchmark_features.py       -Rows 10000000'
Write-Host '  .\run.ps1 05_benchmark_features.py -Gpu  -Rows 10000000'
Write-Host '  .\run.ps1 06_benchmark_target_encoder.py -Rows 2000000'

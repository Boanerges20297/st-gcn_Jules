param(
    [ValidateSet("fortaleza-fast", "fortaleza-full", "state-classic", "state-deep", "state-full")]
    [string]$Preset = "fortaleza-fast",

    [string]$Cutoff = "2025-12-31",
    [string]$TrainStart = "2022-01-01",
    [string]$TrainEnd = "2024-12-31",
    [string]$ValStart = "2025-01-01",
    [string]$ValEnd = "2025-12-31"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "scripts\benchmark_cvli_stochastic_suite.py"

if (-not (Test-Path $python)) {
    throw "Python da .venv não encontrado em $python"
}

if (-not (Test-Path $script)) {
    throw "Script de benchmark não encontrado em $script"
}

$baseArgs = @(
    $script,
    "--cutoff", $Cutoff,
    "--train-start", $TrainStart,
    "--train-end", $TrainEnd,
    "--val-start", $ValStart,
    "--val-end", $ValEnd
)

switch ($Preset) {
    "fortaleza-fast" {
        $argsList = $baseArgs + @(
            "--regions", "fortaleza",
            "--deep-epochs", "4",
            "--classical-models", "zero_baseline", "lag1_baseline", "roll7_baseline", "logit_classifier", "poisson_regressor", "hurdle_logit_poisson",
            "--deep-models", "ShallowGAT", "DeepSTGAT_64"
        )
    }
    "fortaleza-full" {
        $argsList = $baseArgs + @(
            "--regions", "fortaleza",
            "--deep-epochs", "6",
            "--classical-models", "zero_baseline", "lag1_baseline", "roll7_baseline", "logit_classifier", "histgb_classifier", "poisson_regressor", "hurdle_logit_poisson",
            "--deep-models", "ShallowGAT", "DeepSTGAT_64", "PureSTGCN_64", "FortalezaHeteroSTGAT"
        )
    }
    "state-classic" {
        $argsList = $baseArgs + @(
            "--regions", "fortaleza", "rmf", "interior",
            "--deep-epochs", "1",
            "--classical-models", "zero_baseline", "lag1_baseline", "roll7_baseline", "logit_classifier", "histgb_classifier", "poisson_regressor", "hurdle_logit_poisson",
            "--deep-models", "ShallowGAT"
        )
    }
    "state-deep" {
        $argsList = $baseArgs + @(
            "--regions", "fortaleza", "rmf", "interior",
            "--deep-epochs", "6",
            "--classical-models", "logit_classifier",
            "--deep-models", "ShallowGAT", "DeepSTGAT_32", "DeepSTGAT_64", "PureSTGCN_64", "FortalezaHeteroSTGAT"
        )
    }
    "state-full" {
        $argsList = $baseArgs + @(
            "--regions", "fortaleza", "rmf", "interior",
            "--deep-epochs", "6",
            "--classical-models", "zero_baseline", "lag1_baseline", "roll7_baseline", "logit_classifier", "histgb_classifier", "poisson_regressor", "hurdle_logit_poisson",
            "--deep-models", "ShallowGAT", "DeepSTGAT_32", "DeepSTGAT_64", "PureSTGCN_64", "FortalezaHeteroSTGAT"
        )
    }
}

Write-Host ""
Write-Host "CVLI stochastic benchmark preset: $Preset" -ForegroundColor Cyan
Write-Host "Command:" -ForegroundColor Yellow
Write-Host "$python $($argsList -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& $python @argsList

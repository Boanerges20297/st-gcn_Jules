[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',
    [string]$HermesWorkspace = 'E:\Hermes_Workspace',
    [string]$GeminiModel = 'gemini-2.5-flash'
)

$ErrorActionPreference = 'Stop'

$pythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$gatewayScript = Join-Path $ProjectRoot 'powershell\telegram_gemini_gateway.py'
$pidFile = Join-Path $ProjectRoot 'outputs\hermes\chat\telegram_gemini_gateway.pid'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python nao encontrado em $pythonPath"
}

if (-not (Test-Path -LiteralPath $gatewayScript)) {
    throw "Gateway script nao encontrado em $gatewayScript"
}

$hasHermes = $true
if (-not (Test-Path -LiteralPath $HermesWorkspace)) {
    Write-Warning "HermesWorkspace nao encontrado em $HermesWorkspace. O bot do Telegram funcionara em modo stand-alone (sem SOUL.md)."
    $hasHermes = $false
}

if ($hasHermes) {
    Push-Location $HermesWorkspace
    try {
        if ($PSCmdlet.ShouldProcess($HermesWorkspace, 'Parar gateway nativo do Hermes para liberar o bot Telegram')) {
            hermes gateway stop
        }
    }
    finally {
        Pop-Location
    }
}

$arguments = @(
    ('"' + $gatewayScript + '"'),
    '--project-root', ('"' + $ProjectRoot + '"'),
    '--gemini-model', ('"' + $GeminiModel + '"')
)

if ($hasHermes) {
    $arguments += @('--hermes-workspace', ('"' + $HermesWorkspace + '"'))
}

if ($PSCmdlet.ShouldProcess($ProjectRoot, 'Iniciar gateway Telegram Gemini em nova janela')) {
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Normal -PassThru
    New-Item -ItemType Directory -Path (Split-Path $pidFile -Parent) -Force | Out-Null
    [System.IO.File]::WriteAllText($pidFile, $process.Id.ToString(), [System.Text.Encoding]::UTF8)
    Write-Host "Gateway Telegram Gemini iniciado. PID: $($process.Id)"
    Write-Host "PID salvo em: $pidFile"
}
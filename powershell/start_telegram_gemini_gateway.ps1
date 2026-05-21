[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',
    [string]$HermesWorkspace = 'E:\Hermes_Workspace',
    [string]$GeminiModel = 'gemini-2.5-flash'
)

$ErrorActionPreference = 'Stop'

$pythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$gatewayScript = Join-Path $ProjectRoot 'telegram_gemini_gateway.py'
$pidFile = Join-Path $ProjectRoot 'outputs\hermes\chat\telegram_gemini_gateway.pid'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python nao encontrado em $pythonPath"
}

if (-not (Test-Path -LiteralPath $gatewayScript)) {
    throw "Gateway script nao encontrado em $gatewayScript"
}

if (-not (Test-Path -LiteralPath $HermesWorkspace)) {
    throw "HermesWorkspace nao encontrado em $HermesWorkspace. Verifique se o HD externo esta montado e se a pasta existe, ou informe -HermesWorkspace com o caminho correto."
}

Push-Location $HermesWorkspace
try {
    if ($PSCmdlet.ShouldProcess($HermesWorkspace, 'Parar gateway nativo do Hermes para liberar o bot Telegram')) {
        hermes gateway stop
    }
}
finally {
    Pop-Location
}

$arguments = @(
    ('"' + $gatewayScript + '"'),
    '--project-root', ('"' + $ProjectRoot + '"'),
    '--hermes-workspace', ('"' + $HermesWorkspace + '"'),
    '--gemini-model', ('"' + $GeminiModel + '"')
)

if ($PSCmdlet.ShouldProcess($ProjectRoot, 'Iniciar gateway Telegram Gemini em nova janela')) {
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Normal -PassThru
    New-Item -ItemType Directory -Path (Split-Path $pidFile -Parent) -Force | Out-Null
    [System.IO.File]::WriteAllText($pidFile, $process.Id.ToString(), [System.Text.Encoding]::UTF8)
    Write-Host "Gateway Telegram Gemini iniciado. PID: $($process.Id)"
    Write-Host "PID salvo em: $pidFile"
}
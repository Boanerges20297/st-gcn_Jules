[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview'
)

$ErrorActionPreference = 'Stop'

$pidFile = Join-Path $ProjectRoot 'outputs\hermes\chat\telegram_gemini_gateway.pid'

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host 'Nenhum PID salvo para o gateway Telegram Gemini.'
    exit 0
}

$pidText = [System.IO.File]::ReadAllText($pidFile, [System.Text.Encoding]::UTF8).Trim()
if (-not $pidText) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host 'PID vazio removido.'
    exit 0
}

$processId = [int]$pidText

if ($PSCmdlet.ShouldProcess($processId, 'Encerrar gateway Telegram Gemini')) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Gateway Telegram Gemini encerrado. PID: $processId"
}
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',
    [string]$HermesWorkspace = 'E:\Hermes_Workspace',
    [string]$GeminiModel = 'gemini-2.5-flash'
)

$ErrorActionPreference = 'Stop'

$scriptRoot = $PSScriptRoot
if (-not $scriptRoot) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$stopScript = Join-Path $scriptRoot 'stop_telegram_gemini_gateway.ps1'
$startScript = Join-Path $scriptRoot 'start_telegram_gemini_gateway.ps1'

if (-not (Test-Path -LiteralPath $stopScript)) {
    throw "Script de parada nao encontrado em $stopScript"
}

if (-not (Test-Path -LiteralPath $startScript)) {
    throw "Script de inicializacao nao encontrado em $startScript"
}

if ($PSCmdlet.ShouldProcess($ProjectRoot, 'Reiniciar gateway Telegram Gemini')) {
    & powershell -ExecutionPolicy Bypass -File $stopScript -ProjectRoot $ProjectRoot
    & powershell -ExecutionPolicy Bypass -File $startScript -ProjectRoot $ProjectRoot -HermesWorkspace $HermesWorkspace -GeminiModel $GeminiModel
    Write-Host 'Reinicio do gateway Telegram Gemini concluido.'
}
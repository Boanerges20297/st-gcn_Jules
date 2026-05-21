[CmdletBinding()]
param(
    [int]$Limit = 20,
    [string]$Username,
    [string]$EventType
)

$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'powershell\auth_audit_report.ps1'
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Script nao encontrado em $scriptPath"
}

if (-not [string]::IsNullOrWhiteSpace($Username) -and -not [string]::IsNullOrWhiteSpace($EventType)) {
    & $scriptPath -ProjectRoot $PSScriptRoot -Limit $Limit -Username $Username -EventType $EventType
}
elseif (-not [string]::IsNullOrWhiteSpace($Username)) {
    & $scriptPath -ProjectRoot $PSScriptRoot -Limit $Limit -Username $Username
}
elseif (-not [string]::IsNullOrWhiteSpace($EventType)) {
    & $scriptPath -ProjectRoot $PSScriptRoot -Limit $Limit -EventType $EventType
}
else {
    & $scriptPath -ProjectRoot $PSScriptRoot -Limit $Limit
}

[CmdletBinding()]
param(
    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',
    [string]$Username
)

$ErrorActionPreference = 'Stop'

function Convert-SecureStringToPlainText {
    param([Security.SecureString]$SecureValue)

    if ($null -eq $SecureValue) {
        return ''
    }

    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

$pythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python nao encontrado em $pythonPath"
}

if ([string]::IsNullOrWhiteSpace($Username)) {
    $Username = Read-Host 'Usuario do bot'
}

$Username = $Username.Trim()
if ([string]::IsNullOrWhiteSpace($Username)) {
    throw 'Usuario invalido.'
}

$existsCheck = @"
import sys
from manage_telegram_users import ensure_db, user_exists
ensure_db()
print('1' if user_exists(sys.argv[1]) else '0')
"@

Push-Location $ProjectRoot
try {
    $existsRaw = & $pythonPath -c $existsCheck $Username
    if ($LASTEXITCODE -ne 0) {
        throw 'Falha ao validar o usuario no SQLite local.'
    }

    $userExists = (($existsRaw | Select-Object -Last 1).ToString().Trim() -eq '1')
    if ($userExists) {
        Write-Host "Usuario '$Username' encontrado. A senha sera atualizada."
    }
    else {
        Write-Host "Usuario '$Username' nao existe. Ele sera criado."
    }

    $passwordSecure = Read-Host 'Senha' -AsSecureString
    $confirmSecure = Read-Host 'Confirmar senha' -AsSecureString
    $passwordPlain = Convert-SecureStringToPlainText -SecureValue $passwordSecure
    $confirmPlain = Convert-SecureStringToPlainText -SecureValue $confirmSecure

    if ([string]::IsNullOrWhiteSpace($passwordPlain)) {
        throw 'Senha vazia nao e permitida.'
    }

    if ($passwordPlain -ne $confirmPlain) {
        throw 'As senhas nao conferem.'
    }

    $payload = @{
        username = $Username
        password = $passwordPlain
        mode = if ($userExists) { 'set-password' } else { 'add' }
    } | ConvertTo-Json -Compress

    $upsertCode = @"
import os
import json
from manage_telegram_users import ensure_db, add_user, set_password
payload = json.loads(os.environ['TELEGRAM_UPSERT_PAYLOAD'])
ensure_db()
if payload.get('mode') == 'set-password':
    set_password(payload['username'], payload['password'])
else:
    add_user(payload['username'], payload['password'])
"@

    $previousPayload = $env:TELEGRAM_UPSERT_PAYLOAD
    try {
        $env:TELEGRAM_UPSERT_PAYLOAD = $payload
        & $pythonPath -c $upsertCode
        if ($LASTEXITCODE -ne 0) {
            throw 'Falha ao gravar o usuario no SQLite local.'
        }
    }
    finally {
        if ($null -ne $previousPayload) {
            $env:TELEGRAM_UPSERT_PAYLOAD = $previousPayload
        }
        else {
            Remove-Item Env:TELEGRAM_UPSERT_PAYLOAD -ErrorAction SilentlyContinue
        }
    }
}
finally {
    if ($passwordPlain) { $passwordPlain = $null }
    if ($confirmPlain) { $confirmPlain = $null }
    Pop-Location
}

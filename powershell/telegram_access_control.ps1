[CmdletBinding()]
param(
    [ValidateSet('status', 'block-user', 'unblock-user', 'lock-global', 'unlock-global')]
    [string]$Action = 'status',
    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',
    [string]$Username,
    [string]$Reason
)

$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python nao encontrado em $pythonPath"
}

$script = @"
import os
import json
from manage_telegram_users import ensure_db, get_global_lock, set_active, set_global_lock

ensure_db()
payload = json.loads(os.environ['TELEGRAM_ACCESS_CONTROL_PAYLOAD'])
action = payload.get('action', '')
username = payload.get('username', '')
reason = payload.get('reason', '')

if action == 'status':
    lock_state = get_global_lock()
    print(json.dumps({'global_lock': lock_state}, ensure_ascii=False))
elif action == 'block-user':
    set_active(username, False)
    print(json.dumps({'action': action, 'username': username}, ensure_ascii=False))
elif action == 'unblock-user':
    set_active(username, True)
    print(json.dumps({'action': action, 'username': username}, ensure_ascii=False))
elif action == 'lock-global':
    set_global_lock(True, reason)
    print(json.dumps({'action': action, 'reason': reason}, ensure_ascii=False))
elif action == 'unlock-global':
    set_global_lock(False, reason)
    print(json.dumps({'action': action, 'reason': reason}, ensure_ascii=False))
else:
    raise ValueError(f'Ação inválida: {action}')
"@

Push-Location $ProjectRoot
try {
    $payload = @{
        action = $Action
        username = if ([string]::IsNullOrWhiteSpace($Username)) { '' } else { $Username }
        reason = if ([string]::IsNullOrWhiteSpace($Reason)) { '' } else { $Reason }
    } | ConvertTo-Json -Compress

    $previousPayload = $env:TELEGRAM_ACCESS_CONTROL_PAYLOAD
    try {
        $env:TELEGRAM_ACCESS_CONTROL_PAYLOAD = $payload
        & $pythonPath -c $script
        if ($LASTEXITCODE -ne 0) {
            throw 'Falha ao aplicar controle de acesso do Telegram.'
        }
    }
    finally {
        if ($null -ne $previousPayload) {
            $env:TELEGRAM_ACCESS_CONTROL_PAYLOAD = $previousPayload
        }
        else {
            Remove-Item Env:TELEGRAM_ACCESS_CONTROL_PAYLOAD -ErrorAction SilentlyContinue
        }
    }
}
finally {
    Pop-Location
}

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$HermesWorkspace = 'E:\Hermes_Workspace',
    [string]$TelegramUserId = '80086019',
    [switch]$NoRestart
)

$ErrorActionPreference = 'Stop'

function ConvertTo-Hashtable {
    param(
        [Parameter(ValueFromPipeline = $true)]
        $InputObject
    )

    if ($null -eq $InputObject) {
        return $null
    }

    if ($InputObject -is [System.Collections.IDictionary]) {
        $hash = [ordered]@{}
        foreach ($key in $InputObject.Keys) {
            $hash[$key] = ConvertTo-Hashtable -InputObject $InputObject[$key]
        }
        return $hash
    }

    if (($InputObject -isnot [string]) -and ($InputObject -is [System.Collections.IEnumerable])) {
        $items = @()
        foreach ($item in $InputObject) {
            $items += ,(ConvertTo-Hashtable -InputObject $item)
        }
        return $items
    }

    if ($InputObject -is [psobject]) {
        $properties = $InputObject.PSObject.Properties
        if ($properties.Count -gt 0) {
            $hash = [ordered]@{}
            foreach ($property in $properties) {
                $hash[$property.Name] = ConvertTo-Hashtable -InputObject $property.Value
            }
            return $hash
        }
    }

    return $InputObject
}

function Remove-TelegramSessions {
    param(
        [string]$SessionsDir,
        [string]$UserId
    )

    $sessionsIndexPath = Join-Path $SessionsDir 'sessions.json'
    if (-not (Test-Path $sessionsIndexPath)) {
        Write-Warning "Arquivo de sessoes nao encontrado: $sessionsIndexPath"
        return @()
    }

    $raw = Get-Content -Path $sessionsIndexPath -Raw -Encoding UTF8
    $sessionsIndex = if ([string]::IsNullOrWhiteSpace($raw)) {
        [ordered]@{}
    } else {
        ConvertTo-Hashtable -InputObject (ConvertFrom-Json -InputObject $raw)
    }

    $removedSessionIds = New-Object System.Collections.Generic.List[string]
    $remaining = [ordered]@{}

    $entries = @()
    if ($sessionsIndex -is [System.Collections.IDictionary]) {
        $entries = $sessionsIndex.GetEnumerator()
    } else {
        $entries = $sessionsIndex.PSObject.Properties | ForEach-Object {
            [pscustomobject]@{
                Key = $_.Name
                Value = $_.Value
            }
        }
    }

    foreach ($entry in $entries) {
        $sessionKey = [string]$entry.Key
        $sessionValue = $entry.Value
        $origin = $sessionValue.origin
        $isTelegram = $origin.platform -eq 'telegram'
        $matchesUser = ([string]$origin.user_id -eq $UserId) -or ([string]$origin.chat_id -eq $UserId)

        if ($isTelegram -and $matchesUser) {
            $sessionId = [string]$sessionValue.session_id
            if (-not [string]::IsNullOrWhiteSpace($sessionId)) {
                [void]$removedSessionIds.Add($sessionId)
            }
            Write-Host "Removendo mapeamento de sessao Telegram: $sessionKey"
            continue
        }

        $remaining[$sessionKey] = $sessionValue
    }

    if ($PSCmdlet.ShouldProcess($sessionsIndexPath, 'Atualizar sessions.json removendo sessoes Telegram')) {
        $json = $remaining | ConvertTo-Json -Depth 10
        Set-Content -Path $sessionsIndexPath -Value $json -Encoding UTF8
    }

    foreach ($sessionId in $removedSessionIds) {
        foreach ($candidate in @(
            (Join-Path $SessionsDir ("session_{0}.json" -f $sessionId)),
            (Join-Path $SessionsDir ("{0}.jsonl" -f $sessionId))
        )) {
            if (Test-Path $candidate) {
                if ($PSCmdlet.ShouldProcess($candidate, 'Excluir arquivo de sessao Telegram')) {
                    Remove-Item -Path $candidate -Force
                }
            }
        }
    }

    return $removedSessionIds
}

function Restart-HermesGateway {
    param(
        [string]$Workspace
    )

    Push-Location $Workspace
    try {
        if ($PSCmdlet.ShouldProcess($Workspace, 'Parar gateway Hermes')) {
            hermes gateway stop
        }

        $command = "Set-Location -Path '$Workspace'; hermes gateway run"
        if ($PSCmdlet.ShouldProcess($Workspace, 'Iniciar gateway Hermes em nova janela')) {
            Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoExit', '-Command', $command | Out-Null
        }
    }
    finally {
        Pop-Location
    }
}

$sessionsDir = Join-Path $HermesWorkspace '.hermes\sessions'
if (-not (Test-Path $sessionsDir)) {
    throw "Diretorio de sessoes Hermes nao encontrado: $sessionsDir"
}

Write-Host "Hermes workspace: $HermesWorkspace"
Write-Host "Telegram user/chat alvo: $TelegramUserId"

$removed = Remove-TelegramSessions -SessionsDir $sessionsDir -UserId $TelegramUserId

if ($removed.Count -eq 0) {
    Write-Host 'Nenhuma sessao Telegram correspondente foi encontrada.'
} else {
    Write-Host ("Sessoes Telegram removidas: {0}" -f ($removed -join ', '))
}

if ($NoRestart) {
    Write-Host 'Gateway nao reiniciado porque -NoRestart foi informado.'
} else {
    Restart-HermesGateway -Workspace $HermesWorkspace
    Write-Host 'Gateway reiniciado em nova janela do PowerShell.'
}

Write-Host 'Reset concluido.'
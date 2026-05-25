# deploy_telegram_changes.ps1
# Sincroniza os arquivos do gateway Telegram na VPS e recria o container.
#
# Uso manual:
#   powershell -ExecutionPolicy Bypass -File .\powershell\deploy_telegram_changes.ps1
#
# Por padrao, o script le .env:
#   HOST_SSH=root@76.13.121.172
#   PASSWORD_VPS_SSH=<senha>
#
# Se PASSWORD_VPS_SSH existir e o WinSCP estiver instalado, usa SFTP/SSH com senha.
# Se nao existir senha, usa ssh/scp do OpenSSH, assumindo chave SSH configurada.

[CmdletBinding()]
param(
    [string]$RemoteHost = '76.13.121.172',
    [string]$User = 'reportpreview',
    [int]$Port = 22,
    [string]$TargetDir = '/home/reportpreview/apps/report-preview',
    [string]$RemoteOwner = 'reportpreview:reportpreview',
    [string]$ProjectRoot = '',
    [string]$EnvPath = '',
    [string]$WinScpDllPath = 'C:\Program Files (x86)\WinSCP\WinSCPnet.dll',
    [switch]$UseSshTools,
    [switch]$NoRestart
)

$ErrorActionPreference = 'Stop'

function Read-DotEnv {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $values
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*#' -or $line -notmatch '=') {
            continue
        }

        $key, $value = $line.Split('=', 2)
        $key = $key.Trim()
        if ([string]::IsNullOrWhiteSpace($key)) {
            continue
        }

        $values[$key] = $value.Trim().Trim('"').Trim("'")
    }

    return $values
}

function Split-HostSpec {
    param(
        [string]$HostSpec,
        [string]$DefaultUser,
        [string]$DefaultHost
    )

    if ([string]::IsNullOrWhiteSpace($HostSpec)) {
        return @{ User = $DefaultUser; Host = $DefaultHost }
    }

    if ($HostSpec.Contains('@')) {
        $parts = $HostSpec.Split('@', 2)
        return @{ User = $parts[0]; Host = $parts[1] }
    }

    return @{ User = $DefaultUser; Host = $HostSpec }
}

function Invoke-RemoteChecked {
    param(
        [WinSCP.Session]$Session,
        [string]$Command,
        [string]$FailureMessage
    )

    $result = $Session.ExecuteCommand($Command)
    if (-not [string]::IsNullOrWhiteSpace($result.Output)) {
        Write-Host $result.Output.TrimEnd()
    }
    if ($result.ExitCode -ne 0) {
        throw ($FailureMessage + ': ' + $result.ErrorOutput)
    }
}

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}

if (-not $EnvPath) {
    $EnvPath = Join-Path $ProjectRoot '.env'
}

$envData = Read-DotEnv -Path $EnvPath
$hostParts = Split-HostSpec -HostSpec $envData['HOST_SSH'] -DefaultUser $User -DefaultHost $RemoteHost
$User = $hostParts.User
$RemoteHost = $hostParts.Host
$password = $envData['PASSWORD_VPS_SSH']
$remote = ('{0}@{1}' -f $User, $RemoteHost)
$quote = [char]39

$changedFiles = @(
    @{ Local = 'docker-compose.telegram-only.yml'; Remote = "$TargetDir/docker-compose.telegram-only.yml" },
    @{ Local = 'docker\Dockerfile.telegram-gateway'; Remote = "$TargetDir/docker/Dockerfile.telegram-gateway" },
    @{ Local = 'scripts\linux\ask_gemini_with_mempalace.py'; Remote = "$TargetDir/scripts/linux/ask_gemini_with_mempalace.py" },
    @{ Local = 'powershell\telegram_gemini_gateway.py'; Remote = "$TargetDir/powershell/telegram_gemini_gateway.py" }
)

Write-Host '=== DEPLOY TELEGRAM GATEWAY ===' -ForegroundColor Cyan
Write-Host ('Host: {0} | Usuario: {1} | Destino: {2}' -f $RemoteHost, $User, $TargetDir)

foreach ($file in $changedFiles) {
    $localPath = Join-Path $ProjectRoot $file.Local
    if (-not (Test-Path -LiteralPath $localPath)) {
        throw ('Arquivo necessario nao encontrado: {0}' -f $file.Local)
    }
}

$canUseWinScp = (-not $UseSshTools.IsPresent) -and (-not [string]::IsNullOrWhiteSpace($password)) -and (Test-Path -LiteralPath $WinScpDllPath)

if ($canUseWinScp) {
    Write-Host 'Modo: WinSCP SFTP/SSH com senha do .env' -ForegroundColor DarkCyan
    Add-Type -Path $WinScpDllPath

    $sessionOptions = New-Object WinSCP.SessionOptions -Property @{
        Protocol = [WinSCP.Protocol]::Sftp
        HostName = $RemoteHost
        UserName = $User
        Password = $password
        PortNumber = $Port
        GiveUpSecurityAndAcceptAnySshHostKey = $true
    }

    $session = New-Object WinSCP.Session
    try {
        $session.Open($sessionOptions)

        $prepareCommand = 'mkdir -p ' + $quote + $TargetDir + '/docker' + $quote + ' ' + $quote + $TargetDir + '/scripts/linux' + $quote + ' ' + $quote + $TargetDir + '/powershell' + $quote
        Invoke-RemoteChecked -Session $session -Command $prepareCommand -FailureMessage 'Falha ao criar diretorios remotos'

        $transferOptions = New-Object WinSCP.TransferOptions
        $transferOptions.TransferMode = [WinSCP.TransferMode]::Binary

        foreach ($file in $changedFiles) {
            $localPath = Join-Path $ProjectRoot $file.Local
            $result = $session.PutFiles($localPath, $file.Remote, $false, $transferOptions)
            $result.Check()
            Write-Host ('Uploaded: {0}' -f $file.Local)
        }

        if (-not [string]::IsNullOrWhiteSpace($RemoteOwner)) {
            $ownerCommand = 'chown -R ' + $RemoteOwner + ' ' + $quote + $TargetDir + '/docker-compose.telegram-only.yml' + $quote + ' ' + $quote + $TargetDir + '/docker' + $quote + ' ' + $quote + $TargetDir + '/scripts/linux' + $quote + ' ' + $quote + $TargetDir + '/powershell' + $quote
            Invoke-RemoteChecked -Session $session -Command $ownerCommand -FailureMessage 'Falha ao ajustar ownership remoto'
        }

        if (-not $NoRestart.IsPresent) {
            $rebuildCommand = 'cd ' + $quote + $TargetDir + $quote + '; docker compose -f docker-compose.telegram-only.yml up -d --build --force-recreate; docker compose -f docker-compose.telegram-only.yml ps'
            Invoke-RemoteChecked -Session $session -Command $rebuildCommand -FailureMessage 'Falha no rebuild do container'
        }
    }
    finally {
        $session.Dispose()
    }
}
else {
    Write-Host 'Modo: OpenSSH scp/ssh' -ForegroundColor DarkCyan
    if ([string]::IsNullOrWhiteSpace($password)) {
        Write-Host 'PASSWORD_VPS_SSH nao encontrado; usando chave SSH/agente local.' -ForegroundColor Yellow
    }
    elseif (-not (Test-Path -LiteralPath $WinScpDllPath)) {
        Write-Host ('WinSCPnet.dll nao encontrado em: {0}' -f $WinScpDllPath) -ForegroundColor Yellow
    }

    $sshOpts = @('-o', 'StrictHostKeyChecking=no', '-p', [string]$Port)
    $scpOpts = @('-P', [string]$Port, '-o', 'StrictHostKeyChecking=no')

    $prepareCommand = 'mkdir -p ' + $quote + $TargetDir + '/docker' + $quote + ' ' + $quote + $TargetDir + '/scripts/linux' + $quote + ' ' + $quote + $TargetDir + '/powershell' + $quote
    & ssh @sshOpts $remote $prepareCommand
    if ($LASTEXITCODE -ne 0) {
        throw 'Falha ao criar diretorios remotos'
    }

    foreach ($file in $changedFiles) {
        $localPath = Join-Path $ProjectRoot $file.Local
        & scp @scpOpts $localPath ('{0}:{1}' -f $remote, $file.Remote)
        if ($LASTEXITCODE -ne 0) {
            throw ('Falha no scp de {0}' -f $file.Local)
        }
        Write-Host ('Uploaded: {0}' -f $file.Local)
    }

    if (-not [string]::IsNullOrWhiteSpace($RemoteOwner)) {
        $ownerCommand = 'chown -R ' + $RemoteOwner + ' ' + $quote + $TargetDir + '/docker-compose.telegram-only.yml' + $quote + ' ' + $quote + $TargetDir + '/docker' + $quote + ' ' + $quote + $TargetDir + '/scripts/linux' + $quote + ' ' + $quote + $TargetDir + '/powershell' + $quote
        & ssh @sshOpts $remote $ownerCommand
        if ($LASTEXITCODE -ne 0) {
            throw 'Falha ao ajustar ownership remoto'
        }
    }

    if (-not $NoRestart.IsPresent) {
        $rebuildCommand = 'cd ' + $quote + $TargetDir + $quote + '; docker compose -f docker-compose.telegram-only.yml up -d --build --force-recreate; docker compose -f docker-compose.telegram-only.yml ps'
        & ssh @sshOpts $remote $rebuildCommand
        if ($LASTEXITCODE -ne 0) {
            throw 'Falha no rebuild do container'
        }
    }
}

Write-Host ''
Write-Host '=== DEPLOY CONCLUIDO ===' -ForegroundColor Green
Write-Host 'Comando manual:'
Write-Host '  powershell -ExecutionPolicy Bypass -File .\powershell\deploy_telegram_changes.ps1'

# deploy_telegram_changes.ps1
# Envia apenas os arquivos modificados do bot CPRAIO e recria o container.
# Uso: powershell -ExecutionPolicy Bypass -File powershell\deploy_telegram_changes.ps1
param(
    [string]$RemoteHost  = "76.13.121.172",
    [string]$User        = "reportpreview",
    [int]$Port           = 22,
    [string]$TargetDir   = "/home/reportpreview/apps/report-preview",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$remote  = "${User}@${RemoteHost}"
$sshOpts = @("-o", "StrictHostKeyChecking=no", "-p", "$Port")

Write-Host "=== DEPLOY CPRAIO — ARQUIVOS MODIFICADOS ===" -ForegroundColor Cyan
Write-Host "Host: $RemoteHost | Destino: $TargetDir"

# -----------------------------------------------------------------------
# 1. Arquivos e pastas modificados
# -----------------------------------------------------------------------
$changedFiles = @(
    ".hermes.md",
    "docker-compose.telegram-only.yml",
    "docker\Dockerfile.telegram-gateway",
    "scripts\linux\ask_gemini_with_mempalace.py",
    "powershell\telegram_gemini_gateway.py"
)
$changedDirs = @(".mempalace", ".gemini")

# -----------------------------------------------------------------------
# 2. Criar diretórios remotos
# -----------------------------------------------------------------------
Write-Host "`n[1/3] Criando diretorios remotos..." -ForegroundColor Yellow
& ssh @sshOpts $remote "mkdir -p '${TargetDir}/.mempalace' '${TargetDir}/.gemini' '${TargetDir}/docker'"
if ($LASTEXITCODE -ne 0) { throw "Falha ao criar diretorios remotos" }
Write-Host "  OK" -ForegroundColor Green

# -----------------------------------------------------------------------
# 3. Montar tar local e enviar via pipe SSH
# -----------------------------------------------------------------------
Write-Host "`n[2/3] Empacotando e enviando arquivos..." -ForegroundColor Yellow

$tempStage = Join-Path ([System.IO.Path]::GetTempPath()) ("cpraio_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempStage -Force | Out-Null

try {
    foreach ($rel in $changedFiles) {
        $src = Join-Path $ProjectRoot $rel
        $dst = Join-Path $tempStage $rel
        New-Item -ItemType Directory -Path (Split-Path $dst -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Force
        Write-Host "  + $rel"
    }
    foreach ($rel in $changedDirs) {
        $src = Join-Path $ProjectRoot $rel
        if (Test-Path $src) {
            $dst = Join-Path $tempStage $rel
            Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
            Write-Host "  + $rel/"
        }
    }

    $tarFile = Join-Path ([System.IO.Path]::GetTempPath()) "cpraio_changes.tar"
    Push-Location $tempStage
    try {
        & tar -cf $tarFile .
        if ($LASTEXITCODE -ne 0) { throw "Falha ao criar tar" }
    } finally { Pop-Location }

    Write-Host "  Enviando via scp..."
    $remoteArchive = "/tmp/cpraio_changes.tar"
    & scp @("-P", "$Port", "-o", "StrictHostKeyChecking=no") $tarFile "${remote}:${remoteArchive}"
    if ($LASTEXITCODE -ne 0) { throw "Falha no scp" }

    Write-Host "  Extraindo na VPS..."
    & ssh @sshOpts $remote "tar -xf '${remoteArchive}' -C '${TargetDir}'; rm -f '${remoteArchive}'"
    if ($LASTEXITCODE -ne 0) { throw "Falha ao extrair na VPS" }

} finally {
    if (Test-Path $tempStage) { Remove-Item $tempStage -Recurse -Force }
    if (Test-Path $tarFile -ErrorAction SilentlyContinue) { Remove-Item $tarFile -Force }
}

Write-Host "  Upload concluido." -ForegroundColor Green

# -----------------------------------------------------------------------
# 4. Rebuild + restart do container
# -----------------------------------------------------------------------
Write-Host "`n[3/3] Rebuild e restart do telegram-gateway..." -ForegroundColor Yellow
& ssh @sshOpts $remote "cd '${TargetDir}'; docker compose -f docker-compose.telegram-only.yml up -d --build --force-recreate"
if ($LASTEXITCODE -ne 0) { throw "Falha no rebuild do container" }

Write-Host "`n=== STATUS ===" -ForegroundColor Cyan
& ssh @sshOpts $remote "cd '${TargetDir}'; docker compose -f docker-compose.telegram-only.yml ps"

Write-Host "`n=== DEPLOY CONCLUIDO ===" -ForegroundColor Green
Write-Host "Para ver logs: ssh ${User}@${RemoteHost}"
Write-Host "  cd ${TargetDir}"
Write-Host "  docker compose -f docker-compose.telegram-only.yml logs -f telegram-gateway"

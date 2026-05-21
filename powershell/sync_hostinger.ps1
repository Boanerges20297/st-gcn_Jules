[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [Alias('Host')]
    [string]$RemoteHost,

    [string]$User = 'reportpreview',

    [int]$Port = 22,

    [string]$TargetDir = '/home/reportpreview/apps/report-preview',

    [string]$RemoteOwner = 'reportpreview:reportpreview',

    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',

    [string[]]$Groups = @('core', 'artifacts', 'data'),

    [switch]$DryRun,

    [switch]$SkipSshPrepare
)

$ErrorActionPreference = 'Stop'

function Assert-CommandAvailable {
    param([string]$Name)

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Comando obrigatorio nao encontrado no PATH: $Name"
    }
    return $cmd
}

function New-CleanDirectory {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}

function Copy-RelativePath {
    param(
        [string]$Root,
        [string]$RelativePath,
        [string]$StageRoot
    )

    $source = Join-Path $Root $RelativePath
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Warning "Caminho nao encontrado e sera ignorado: $RelativePath"
        return
    }

    $destination = Join-Path $StageRoot $RelativePath
    $destinationParent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
}

function Remove-StagePath {
    param(
        [string]$StageRoot,
        [string]$RelativePath
    )

    $target = Join-Path $StageRoot $RelativePath
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

function Get-GroupPaths {
    param([string]$Group)

    switch ($Group.ToLowerInvariant()) {
        'core' {
            return @(
                'app.py',
                'Dockerfile',
                'docker-compose.yml',
                'docker-compose.hostinger.yml',
                'requirements.txt',
                '.env.hostinger.example',
                '.mempalace.md',
                '.hermes.md',
                'README.md',
                'CHECKLIST_DEPLOY_HOSTINGER.md',
                'IMPLEMENTACAO_NUVEM_HOSTINGER.md',
                'src',
                'templates',
                'static',
                'config',
                'docker',
                'docs',
                'scripts',
                'powershell'
            )
        }
        'artifacts' {
            return @()
        }
        'telegram_artifacts' {
            return @()
        }
        'outputs' {
            return @()
        }
        'models' {
            return @()
        }
        'static_export' {
            return @(
                'static_export'
            )
        }
        'data' {
            return @(
                'data'
            )
        }
        default {
            throw "Grupo nao suportado: $Group"
        }
    }
}

function Get-NormalizedGroups {
    param([string[]]$InputGroups)

    $normalized = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $InputGroups) {
        if ([string]::IsNullOrWhiteSpace($entry)) {
            continue
        }

        foreach ($part in ($entry -split ',')) {
            $value = $part.Trim().ToLowerInvariant()
            if (-not [string]::IsNullOrWhiteSpace($value)) {
                switch ($value) {
                    'artifacts' {
                        [void]$normalized.Add('outputs')
                        [void]$normalized.Add('models')
                        [void]$normalized.Add('static_export')
                    }
                    default {
                        [void]$normalized.Add($value)
                    }
                }
            }
        }
    }

    return @($normalized | Select-Object -Unique)
}

function Get-DirectCopySpec {
    param([string]$Group)

    switch ($Group.ToLowerInvariant()) {
        'outputs' {
            return @{
                Source = 'outputs'
                RemoteParent = "$TargetDir/"
                RemoteExtractRoot = "$TargetDir"
                Mode = 'directory'
            }
        }
        'models' {
            return @{
                Source = 'models'
                RemoteParent = "$TargetDir/"
                RemoteExtractRoot = "$TargetDir"
                Mode = 'directory'
            }
        }
        'static_export' {
            return @{
                Source = 'static_export'
                RemoteParent = "$TargetDir/"
                RemoteExtractRoot = "$TargetDir"
                Mode = 'directory'
            }
        }
        'telegram_artifacts' {
            return @{
                Source = 'outputs\hermes'
                RemoteParent = "$TargetDir/outputs/"
                RemoteExtractRoot = "$TargetDir"
                Mode = 'directory'
            }
        }
        default {
            return $null
        }
    }
}

function Get-ProjectRelativePath {
    param(
        [string]$Root,
        [string]$Path
    )

    $rootUri = [System.Uri]((Resolve-Path -LiteralPath $Root).Path + [System.IO.Path]::DirectorySeparatorChar)
    $pathUri = [System.Uri](Resolve-Path -LiteralPath $Path).Path
    $relativeUri = $rootUri.MakeRelativeUri($pathUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function New-GroupArchive {
    param(
        [string]$Group,
        [string]$Root,
        [string]$Workspace
    )

    $stageRoot = Join-Path $Workspace ("stage_" + $Group)
    New-CleanDirectory -Path $stageRoot

    foreach ($relativePath in (Get-GroupPaths -Group $Group)) {
        Copy-RelativePath -Root $Root -RelativePath $relativePath -StageRoot $stageRoot
    }

    # Podas de runtime para evitar volume e ruido desnecessarios.
    Remove-StagePath -StageRoot $stageRoot -RelativePath 'outputs\mempalace\chat\history'
    Remove-StagePath -StageRoot $stageRoot -RelativePath 'outputs\hermes\chat\history'
    Remove-StagePath -StageRoot $stageRoot -RelativePath 'data\backup'
    Remove-StagePath -StageRoot $stageRoot -RelativePath 'scripts\linux\__pycache__'
    Remove-StagePath -StageRoot $stageRoot -RelativePath 'powershell\__pycache__'

    $archivePath = Join-Path $Workspace ("report-preview_{0}.tar" -f $Group)
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }

    Push-Location $stageRoot
    try {
        & tar -cf $archivePath .
        if ($LASTEXITCODE -ne 0) {
            throw "Falha ao criar archive TAR do grupo $Group"
        }
    }
    finally {
        Pop-Location
    }

    return $archivePath
}

Assert-CommandAvailable -Name 'ssh' | Out-Null
Assert-CommandAvailable -Name 'scp' | Out-Null
Assert-CommandAvailable -Name 'tar' | Out-Null

$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$remote = "$User@$RemoteHost"
$workspace = Join-Path ([System.IO.Path]::GetTempPath()) ('reportpreview_sync_' + [System.Guid]::NewGuid().ToString('N'))
$Groups = Get-NormalizedGroups -InputGroups $Groups
New-Item -ItemType Directory -Path $workspace -Force | Out-Null

try {
    if (-not $SkipSshPrepare) {
        $prepareCommand = "mkdir -p '$TargetDir' '$TargetDir/data' '$TargetDir/models' '$TargetDir/outputs' '$TargetDir/logs' '$TargetDir/static_export' /srv/reportpreview/sync"
        Write-Host "[PREP] $prepareCommand"
        if (-not $DryRun) {
            & ssh -p $Port $remote $prepareCommand
            if ($LASTEXITCODE -ne 0) {
                throw 'Falha ao preparar diretorios remotos via SSH'
            }
        }
    }

    foreach ($group in $Groups) {
        $normalizedGroup = $group.ToLowerInvariant()

        $directCopy = Get-DirectCopySpec -Group $normalizedGroup
        if ($null -ne $directCopy) {
            $sourcePath = Join-Path $root $directCopy.Source
            if (-not (Test-Path -LiteralPath $sourcePath)) {
                Write-Warning "Caminho nao encontrado e sera ignorado: $($directCopy.Source)"
                continue
            }

            $relativeSourcePath = Get-ProjectRelativePath -Root $root -Path $sourcePath
            $archiveSourcePath = $relativeSourcePath
            $remoteExtractRoot = $directCopy.RemoteExtractRoot.TrimEnd('/')
            $remotePrepare = "mkdir -p '$($directCopy.RemoteParent.TrimEnd('/'))'"
            $remoteStreamExtract = "$remotePrepare && tar -xf - -C '$remoteExtractRoot'"

            Write-Host "============================================================"
            Write-Host "Grupo: $normalizedGroup"
            Write-Host "Modo: tar-stream"
            Write-Host "Origem local: $sourcePath"
            Write-Host "Origem TAR relativa: $archiveSourcePath"
            Write-Host ("Diretorio remoto preparado: {0}" -f $directCopy.RemoteParent)
            Write-Host "Extract remoto: $remoteStreamExtract"
            Write-Host "============================================================"

            if ($DryRun) {
                continue
            }

            if ($PSCmdlet.ShouldProcess($remote, "Enviar diretorio do grupo $normalizedGroup via tar-stream")) {
                Push-Location $root
                try {
                    & tar -cf - $archiveSourcePath | & ssh -p $Port $remote $remoteStreamExtract
                }
                finally {
                    Pop-Location
                }
                if ($LASTEXITCODE -ne 0) {
                    throw "Falha ao enviar diretorio do grupo $normalizedGroup via tar-stream"
                }
            }

            continue
        }

        $archivePath = New-GroupArchive -Group $normalizedGroup -Root $root -Workspace $workspace
        $remoteArchive = "/srv/reportpreview/sync/$(Split-Path -Leaf $archivePath)"
        $remoteExtract = "tar -xf '$remoteArchive' -C '$TargetDir'"
        if ($User -eq 'root' -and -not [string]::IsNullOrWhiteSpace($RemoteOwner)) {
            $remoteExtract += " && chown -R '$RemoteOwner' '$TargetDir'"
        }
        $remoteExtract += " && rm -f '$remoteArchive'"

        Write-Host "============================================================"
        Write-Host "Grupo: $normalizedGroup"
        Write-Host "Archive local: $archivePath"
        Write-Host ("Upload destino: {0}:{1}" -f $remote, $remoteArchive)
        Write-Host "Extract remoto: $remoteExtract"
        Write-Host "============================================================"

        if ($DryRun) {
            continue
        }

        if ($PSCmdlet.ShouldProcess(("{0}:{1}" -f $remote, $remoteArchive), "Enviar archive do grupo $normalizedGroup")) {
            & scp -P $Port $archivePath "${remote}:$remoteArchive"
            if ($LASTEXITCODE -ne 0) {
                throw "Falha ao enviar archive do grupo $normalizedGroup via SCP"
            }
        }

        if ($PSCmdlet.ShouldProcess($remote, "Extrair archive do grupo $normalizedGroup no destino")) {
            & ssh -p $Port $remote $remoteExtract
            if ($LASTEXITCODE -ne 0) {
                throw "Falha ao extrair archive do grupo $normalizedGroup na VPS"
            }
        }
    }

    Write-Host 'Sync concluido.'
    Write-Host "Host: $RemoteHost"
    Write-Host "Usuario: $User"
    Write-Host "Destino: $TargetDir"
    Write-Host ("Grupos: " + ($Groups -join ', '))
    Write-Host ("DryRun: " + [string]$DryRun.IsPresent)
}
finally {
    if (Test-Path -LiteralPath $workspace) {
        Remove-Item -LiteralPath $workspace -Recurse -Force
    }
}
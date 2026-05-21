[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$HermesWorkspace = 'E:\Hermes_Workspace'
)

$ErrorActionPreference = 'Stop'

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

Write-Host "Hermes workspace: $HermesWorkspace"
Restart-HermesGateway -Workspace $HermesWorkspace
Write-Host 'Reinicio do gateway solicitado.'
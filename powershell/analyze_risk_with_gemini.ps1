[CmdletBinding()]
param(
    [ValidateSet('fortaleza', 'rmf', 'interior', 'geral')]
    [string]$Scope = 'fortaleza',

    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',

    [string]$GeminiModel = 'gemini-2.5-flash',

    [switch]$RegenerateArtifacts
)

$ErrorActionPreference = 'Stop'

function Get-ScopeConfig {
    param([string]$RequestedScope)

    switch ($RequestedScope) {
        'fortaleza' {
            return @{
                Csv = 'risk_fortaleza_latest.csv'
                Label = 'Fortaleza'
                Ranking = 'top 30 bairros de Fortaleza'
            }
        }
        'rmf' {
            return @{
                Csv = 'risk_rmf_latest.csv'
                Label = 'RMF'
                Ranking = 'top 20 cidades da RMF'
            }
        }
        'interior' {
            return @{
                Csv = 'risk_interior_latest.csv'
                Label = 'Interior'
                Ranking = 'top 30 cidades do Interior'
            }
        }
        default {
            return @{
                Csv = 'risk_snapshot_latest.csv'
                Label = 'Geral'
                Ranking = 'top 30 cidades do ranking geral'
            }
        }
    }
}

function Get-FileText {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Arquivo nao encontrado: $Path"
    }

    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Get-CsvExcerpt {
    param([string]$Path)

    $columns = @(
        'rank',
        'name',
        'risk_score',
        'risk_level',
        'confidence_pct',
        'expressiveness_pct',
        'top_driver_1',
        'top_driver_2',
        'leitura_rapida_gestor',
        'por_que_importa_gestor',
        'proxima_acao_gestor'
    )

    $rows = Import-Csv -LiteralPath $Path | Select-Object -First 15 -Property $columns
    return (($rows | ConvertTo-Csv -NoTypeInformation) -join [Environment]::NewLine)
}

function Get-ConvergenceCsvExcerpt {
    param([string]$Path)

    $columns = @(
        'data',
        'cidade',
        'bairro',
        'tipo_evento',
        'tipo',
        'nature',
        'qtd_mortes',
        'clima'
    )

    $rows = Import-Csv -LiteralPath $Path | Select-Object -First 120 -Property $columns
    return (($rows | ConvertTo-Csv -NoTypeInformation) -join [Environment]::NewLine)
}

function Invoke-GeminiAnalysis {
    param(
        [string]$PromptText,
        [string]$WorkingDirectory,
        [string]$ModelName
    )

    Push-Location $WorkingDirectory
    try {
        $instruction = 'Analise o contexto recebido via stdin e siga estritamente as instrucoes fornecidas. Use apenas os dados recebidos.'
        $promptPath = [System.IO.Path]::GetTempFileName()
        $stdoutPath = [System.IO.Path]::GetTempFileName()
        $stderrPath = [System.IO.Path]::GetTempFileName()
        $geminiSource = (Get-Command gemini -ErrorAction Stop).Source
        $geminiCmdPath = [System.IO.Path]::ChangeExtension($geminiSource, '.cmd')

        if (-not (Test-Path -LiteralPath $geminiCmdPath)) {
            throw "Nao foi possivel localizar gemini.cmd a partir de $geminiSource"
        }

        [System.IO.File]::WriteAllText($promptPath, $PromptText, [System.Text.Encoding]::UTF8)

        function Invoke-GeminiAttempt {
            param(
                [string]$ModelArg,
                [string]$GeminiCmd,
                [string]$PromptFile,
                [string]$StdoutFile,
                [string]$StderrFile,
                [string]$InstructionText
            )

            $modelSegment = ''
            if (-not [string]::IsNullOrWhiteSpace($ModelArg)) {
                $modelSegment = ' -m "' + $ModelArg + '"'
            }

            $cmdLine = '"' + $GeminiCmd + '" --skip-trust --output-format text' + $modelSegment + ' -p "' + $InstructionText + '" < "' + $PromptFile + '" 1> "' + $StdoutFile + '" 2> "' + $StderrFile + '"'
            cmd.exe /d /c $cmdLine | Out-Null
            return $LASTEXITCODE
        }

        try {
            $exitCode = Invoke-GeminiAttempt -ModelArg $ModelName -GeminiCmd $geminiCmdPath -PromptFile $promptPath -StdoutFile $stdoutPath -StderrFile $stderrPath -InstructionText $instruction
            $stdoutText = [System.IO.File]::ReadAllText($stdoutPath, [System.Text.Encoding]::UTF8)
            $stderrText = [System.IO.File]::ReadAllText($stderrPath, [System.Text.Encoding]::UTF8)

            if ($exitCode -ne 0) {
                $fallbackExitCode = Invoke-GeminiAttempt -ModelArg '' -GeminiCmd $geminiCmdPath -PromptFile $promptPath -StdoutFile $stdoutPath -StderrFile $stderrPath -InstructionText $instruction
                $fallbackStdout = [System.IO.File]::ReadAllText($stdoutPath, [System.Text.Encoding]::UTF8)
                $fallbackStderr = [System.IO.File]::ReadAllText($stderrPath, [System.Text.Encoding]::UTF8)

                if ($fallbackExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($fallbackStdout)) {
                    return @{
                        Text = $fallbackStdout.Trim()
                        ModelUsed = 'default-cli'
                    }
                }

                throw "Gemini CLI falhou com o modelo '$ModelName' e tambem no fallback padrao.`nSTDERR inicial:`n$stderrText`nSTDERR fallback:`n$fallbackStderr"
            }
        }
        finally {
            Remove-Item $promptPath, $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
        }

        if ([string]::IsNullOrWhiteSpace($stdoutText)) {
            throw "Gemini CLI nao retornou conteudo util. STDERR:`n$stderrText"
        }

        return @{
            Text = $stdoutText.Trim()
            ModelUsed = $ModelName
        }
    }
    finally {
        Pop-Location
    }
}

$scopeConfig = Get-ScopeConfig -RequestedScope $Scope
$outputsDir = Join-Path $ProjectRoot 'outputs\hermes'
$historyDir = Join-Path $outputsDir 'history'
$briefPath = Join-Path $outputsDir 'risk_brief_latest.md'
$csvPath = Join-Path $outputsDir $scopeConfig.Csv
$convergenceCsvPath = Join-Path $outputsDir 'dados_status_enriquecido_14d_latest.csv'

if ($RegenerateArtifacts) {
    Push-Location $ProjectRoot
    try {
        & '.\.venv\Scripts\python.exe' -c "import os; from src.core.orchestrator import StateOrchestrator; root=os.getcwd(); StateOrchestrator(root).get_combined_risk(); print('ok')"
        if ($LASTEXITCODE -ne 0) {
            throw 'Falha ao regenerar artefatos Hermes.'
        }
    }
    finally {
        Pop-Location
    }
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$analysisLatestPath = Join-Path $outputsDir ("gemini_analysis_{0}_latest.md" -f $Scope)
$analysisHistoryPath = Join-Path $historyDir ("gemini_analysis_{0}_{1}.md" -f $Scope, $timestamp)
$csvHistoryPath = Join-Path $historyDir ("gemini_input_{0}_{1}.csv" -f $Scope, $timestamp)
$convergenceCsvHistoryPath = Join-Path $historyDir ("gemini_convergence_input_{0}_{1}.csv" -f $Scope, $timestamp)

New-Item -ItemType Directory -Path $historyDir -Force | Out-Null

$briefText = Get-FileText -Path $briefPath
$csvText = Get-FileText -Path $csvPath
$csvExcerpt = Get-CsvExcerpt -Path $csvPath
$convergenceCsvText = Get-FileText -Path $convergenceCsvPath
$convergenceCsvExcerpt = Get-ConvergenceCsvExcerpt -Path $convergenceCsvPath

$prompt = @"
Voce esta analisando saidas do projeto Report Preview para apoio a decisao operacional.

Regras obrigatorias:
- responder em Portugues do Brasil;
- usar tom tecnico, objetivo e gerencial;
- nao inventar bairros, cidades, rankings ou causas fora do material recebido;
- citar apenas nomes, padroes e sinais que estejam presentes no brief ou no CSV recebido;
- identificar padroes, anomalias, concentracao territorial e sinais de prioridade operacional;
- comparar o ranking previsto com os eventos enriquecidos dos ultimos 14 dias e explicitar se ha convergencia, divergencia ou evidencia inconclusiva;
- quando houver limite de confianca, deixar isso explicito;
- encerrar com recomendacoes praticas de verificacao.

Escopo da analise: $($scopeConfig.Label)
Ranking esperado: $($scopeConfig.Ranking)

Estrutura obrigatoria da resposta:
1. Dados ate
2. Leitura rapida
3. Padroes observados
4. Convergencia com dados_status_ENRIQUECIDO (ultimos 14 dias)
5. Pontos de atencao e limites
6. Recomendacoes operacionais

BRIEF HERMES:
$briefText

EXTRATO DO CSV UTILIZADO ($($scopeConfig.Csv), top 15 linhas e colunas essenciais):
$csvExcerpt

CSV DE CONVERGENCIA INDEPENDENTE (dados_status_enriquecido_14d_latest.csv, ate 120 linhas e colunas essenciais):
$convergenceCsvExcerpt
"@

$analysisResult = Invoke-GeminiAnalysis -PromptText $prompt -WorkingDirectory $ProjectRoot -ModelName $GeminiModel
$analysisText = $analysisResult.Text
$modelUsed = $analysisResult.ModelUsed

$header = @(
    "# Analise Gemini CLI - $($scopeConfig.Label)",
    '',
    "Gerado em: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "Escopo: $($scopeConfig.Label)",
    "Modelo Gemini CLI: $modelUsed",
    "CSV fonte: outputs/hermes/$($scopeConfig.Csv)",
    "Historico CSV: outputs/hermes/history/$(Split-Path -Leaf $csvHistoryPath)",
    "CSV convergencia: outputs/hermes/dados_status_enriquecido_14d_latest.csv",
    "Historico convergencia: outputs/hermes/history/$(Split-Path -Leaf $convergenceCsvHistoryPath)",
    ''
) -join [Environment]::NewLine

$finalAnalysis = $header + [Environment]::NewLine + $analysisText.Trim() + [Environment]::NewLine

[System.IO.File]::WriteAllText($analysisLatestPath, $finalAnalysis, [System.Text.Encoding]::UTF8)
[System.IO.File]::WriteAllText($analysisHistoryPath, $finalAnalysis, [System.Text.Encoding]::UTF8)
[System.IO.File]::WriteAllText($csvHistoryPath, $csvText, [System.Text.Encoding]::UTF8)
[System.IO.File]::WriteAllText($convergenceCsvHistoryPath, $convergenceCsvText, [System.Text.Encoding]::UTF8)

Write-Host "Analise salva em: $analysisLatestPath"
Write-Host "Historico da analise: $analysisHistoryPath"
Write-Host "Snapshot do CSV: $csvHistoryPath"
Write-Host "Snapshot do CSV de convergencia: $convergenceCsvHistoryPath"
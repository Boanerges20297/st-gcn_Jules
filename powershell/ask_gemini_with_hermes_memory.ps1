[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Query,

    [ValidateSet('fortaleza', 'rmf', 'interior', 'geral')]
    [string]$Scope = 'fortaleza',

    [string]$ProjectRoot = 'C:\Users\Boanerges\Desktop\Projetos\Report Preview',

    [string]$HermesWorkspace = 'E:\Hermes_Workspace',

    [string]$GeminiModel = 'gemini-2.5-flash'
)

$ErrorActionPreference = 'Stop'

function Get-ScopeConfig {
    param([string]$RequestedScope)

    switch ($RequestedScope) {
        'fortaleza' {
            return @{ Csv = 'risk_fortaleza_latest.csv'; Label = 'Fortaleza' }
        }
        'rmf' {
            return @{ Csv = 'risk_rmf_latest.csv'; Label = 'RMF' }
        }
        'interior' {
            return @{ Csv = 'risk_interior_latest.csv'; Label = 'Interior' }
        }
        default {
            return @{ Csv = 'risk_snapshot_latest.csv'; Label = 'Geral' }
        }
    }
}

function Read-OptionalFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return ''
    }

    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Normalize-Text {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ''
    }

    $normalized = $Value.Normalize([Text.NormalizationForm]::FormD)
    $builder = New-Object System.Text.StringBuilder
    foreach ($char in $normalized.ToCharArray()) {
        if ([Globalization.CharUnicodeInfo]::GetUnicodeCategory($char) -ne [Globalization.UnicodeCategory]::NonSpacingMark) {
            [void]$builder.Append($char)
        }
    }

    return $builder.ToString().ToUpperInvariant().Trim()
}

function Get-QueryTerms {
    param([string]$Value)

    $normalized = Normalize-Text $Value
    if ([string]::IsNullOrWhiteSpace($normalized)) {
        return @()
    }

    $stopWords = @(
        'A','O','AS','OS','DE','DA','DO','DAS','DOS','E','EM','NO','NA','NOS','NAS','UM','UMA','UNS','UMAS',
        'POR','PARA','COM','SEM','SOBRE','ULTIMOS','ULTIMAS','DIAS','DIA','MES','MESes','ANO','ANOS',
        'ANALISE','TATICA','OPERACIONAL','EVENTOS','OBSERVADOS','BASE','DADOS','DEPENDENDO','SNAPSHOT','HERMES',
        'FACA','DIGA','PRINCIPAIS','FOCOS','PADROES','SE','ESTIVER','USANDO','APENAS','ESSE','CSV','DEIXE','ISSO','EXPLICITO',
        'ME','DE','DOS','DAS','QUAL','QUAIS','COMO','ONDE','PORQUE','POR','QUE','BAIRRO','BAIRROS','CIDADE','CIDADES'
    )

    $tokens = $normalized -split '[^A-Z0-9]+'
    return @($tokens | Where-Object { $_.Length -ge 3 -and $stopWords -notcontains $_ } | Select-Object -Unique)
}

function Get-QuerySpecificContext {
    param(
        [string]$Query,
        [string]$ScopeCsvPath,
        [string]$TacticalCsvPath,
        [string]$RequestedScope
    )

    $queryNormalized = Normalize-Text $Query
    $queryTerms = Get-QueryTerms $Query
    $scopeMatches = @()
    $tacticalMatches = @()

    if (Test-Path -LiteralPath $ScopeCsvPath) {
        $scopeRows = @(Import-Csv -LiteralPath $ScopeCsvPath)
        $scopeMatches = @(
            $scopeRows | Where-Object {
                $nameNormalized = Normalize-Text $_.name
                $direct = $queryNormalized.Length -ge 4 -and $nameNormalized.Contains($queryNormalized)
                $termMatch = $queryTerms.Count -gt 0 -and ($queryTerms | Where-Object { $nameNormalized.Contains($_) }).Count -gt 0
                $direct -or $termMatch
            } | Select-Object -First 8 -Property rank, name, risk_score, risk_level, confidence_pct, top_driver_1, leitura_rapida_gestor, por_que_importa_gestor
        )
    }

    if (Test-Path -LiteralPath $TacticalCsvPath) {
        $rows = @(Import-Csv -LiteralPath $TacticalCsvPath)
        $scopeRows = $rows
        if ($RequestedScope -eq 'fortaleza') {
            $scopeRows = @($rows | Where-Object { (Normalize-Text $_.cidade) -eq 'FORTALEZA' })
        }

        $tacticalMatches = @(
            $scopeRows | Where-Object {
                $cidadeNormalized = Normalize-Text $_.cidade
                $bairroNormalized = Normalize-Text $_.bairro
                $nameNormalized = Normalize-Text $_.name
                $haystack = @($cidadeNormalized, $bairroNormalized, $nameNormalized) -join ' '
                $direct = $queryNormalized.Length -ge 4 -and $haystack.Contains($queryNormalized)
                $termMatch = $queryTerms.Count -gt 0 -and ($queryTerms | Where-Object { $haystack.Contains($_) }).Count -gt 0
                $direct -or $termMatch
            }
        )
    }

    $lines = @(
        "Pergunta normalizada: $queryNormalized",
        "Termos extraidos: $($queryTerms -join ', ')",
        ''
    )

    if ($scopeMatches.Count -gt 0) {
        $lines += 'Correspondencias no CSV oficial do escopo:'
        $lines += (($scopeMatches | ConvertTo-Csv -NoTypeInformation) -join [Environment]::NewLine)
        $lines += ''
    }
    else {
        $lines += 'Correspondencias no CSV oficial do escopo: nenhuma correspondencia direta.'
        $lines += ''
    }

    if ($tacticalMatches.Count -gt 0) {
        $matchGroups = @($tacticalMatches | Group-Object bairro | Where-Object { -not [string]::IsNullOrWhiteSpace($_.Name) } | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { "- Bairro: $($_.Name) | registros: $($_.Count)" })
        $eventGroups = @($tacticalMatches | Group-Object tipo_evento | Where-Object { -not [string]::IsNullOrWhiteSpace($_.Name) } | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { "- Tipo evento: $($_.Name) | registros: $($_.Count)" })
        $sample = @($tacticalMatches | Select-Object -First 30 -Property data, cidade, bairro, tipo_evento, nature, qtd_mortes, clima)
        $lines += "Correspondencias no CSV tatico 14d: $($tacticalMatches.Count) registros"
        $lines += 'Bairros/cidades relacionados encontrados:'
        $lines += ($matchGroups -join [Environment]::NewLine)
        $lines += ''
        $lines += 'Padroes operacionais nas correspondencias:'
        $lines += ($eventGroups -join [Environment]::NewLine)
        $lines += ''
        $lines += 'Extrato das correspondencias especificas:'
        $lines += (($sample | ConvertTo-Csv -NoTypeInformation) -join [Environment]::NewLine)
    }
    else {
        $lines += 'Correspondencias no CSV tatico 14d: nenhuma correspondencia direta.'
        $lines += 'Mesmo sem match direto, ainda existe contexto tatico agregado do escopo e ele deve ser usado para responder de forma util, sem negativa vazia.'
    }

    return ($lines -join [Environment]::NewLine).Trim()
}

function Get-CsvExcerpt {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return ''
    }

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

    $rows = Import-Csv -LiteralPath $Path | Select-Object -First 12 -Property $columns
    return (($rows | ConvertTo-Csv -NoTypeInformation) -join [Environment]::NewLine)
}

function Get-Tactical14dContext {
    param(
        [string]$Path,
        [string]$RequestedScope,
        [string]$ScopeCsvPath
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return "CSV tatico de 14 dias indisponivel."
    }

    $rows = @(Import-Csv -LiteralPath $Path)
    if ($rows.Count -eq 0) {
        return "CSV tatico de 14 dias sem registros."
    }

    $scopeRows = $rows
    if ($RequestedScope -eq 'fortaleza') {
        $scopeRows = @($rows | Where-Object { (Normalize-Text $_.cidade) -eq 'FORTALEZA' })
    }
    elseif ($RequestedScope -in @('rmf', 'interior')) {
        $scopeNames = @()
        if (Test-Path -LiteralPath $ScopeCsvPath) {
            $scopeNames = @(Import-Csv -LiteralPath $ScopeCsvPath | ForEach-Object { Normalize-Text $_.name } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
        }
        if ($scopeNames.Count -gt 0) {
            $scopeRows = @($rows | Where-Object { $scopeNames -contains (Normalize-Text $_.cidade) })
        }
    }

    if ($scopeRows.Count -eq 0) {
        $scopeRows = $rows
    }

    $datedRows = @($scopeRows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.data) } | ForEach-Object {
        $_ | Add-Member -NotePropertyName data_dt -NotePropertyValue ([datetime]::Parse($_.data)) -PassThru
    })

    if ($datedRows.Count -eq 0) {
        $datedRows = $scopeRows
    }

    $topCities = @($scopeRows | Group-Object cidade | Sort-Object Count -Descending | Select-Object -First 5 | ForEach-Object { "- Cidade: $($_.Name) | registros: $($_.Count)" })
    $topBairros = @($scopeRows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.bairro) } | Group-Object bairro | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { "- Bairro: $($_.Name) | registros: $($_.Count)" })
    $topEventos = @($scopeRows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.tipo_evento) } | Group-Object tipo_evento | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { "- Tipo evento: $($_.Name) | registros: $($_.Count)" })
    $recentExcerpt = @($scopeRows | Select-Object -First 80 -Property data, cidade, bairro, tipo_evento, tipo, nature, qtd_mortes, clima)
    $recentCsv = if ($recentExcerpt.Count -gt 0) { (($recentExcerpt | ConvertTo-Csv -NoTypeInformation) -join [Environment]::NewLine) } else { '' }

    $dateMin = ''
    $dateMax = ''
    if ($datedRows.Count -gt 0 -and $datedRows[0].PSObject.Properties.Name -contains 'data_dt') {
        $dateMin = ($datedRows | Measure-Object -Property data_dt -Minimum).Minimum.ToString('yyyy-MM-dd')
        $dateMax = ($datedRows | Measure-Object -Property data_dt -Maximum).Maximum.ToString('yyyy-MM-dd')
    }

    $lines = @(
        "Fonte complementar independente: outputs/hermes/dados_status_enriquecido_14d_latest.csv",
        "Escopo tatico considerado: $RequestedScope",
        "Registros considerados: $($scopeRows.Count)",
        "Janela observada: $dateMin ate $dateMax",
        '',
        'Top cidades por registros nos ultimos 14 dias:',
        ($topCities -join [Environment]::NewLine),
        '',
        'Top bairros por registros nos ultimos 14 dias:',
        ($topBairros -join [Environment]::NewLine),
        '',
        'Top tipos de evento nos ultimos 14 dias:',
        ($topEventos -join [Environment]::NewLine),
        '',
        'Extrato operacional do CSV tatico 14d:',
        $recentCsv
    )

    return (($lines | Where-Object { $_ -ne $null }) -join [Environment]::NewLine).Trim()
}

function Invoke-GeminiText {
    param(
        [string]$PromptText,
        [string]$WorkingDirectory,
        [string]$ModelName
    )

    Push-Location $WorkingDirectory
    try {
        $instruction = 'Responda usando apenas o contexto recebido via stdin. Seja objetivo, analitico e fiel aos artefatos recebidos.'
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
                    return @{ Text = $fallbackStdout.Trim(); ModelUsed = 'default-cli'; Stderr = $fallbackStderr }
                }

                throw "Gemini CLI falhou com '$ModelName' e no fallback.`nSTDERR inicial:`n$stderrText`nSTDERR fallback:`n$fallbackStderr"
            }
        }
        finally {
            Remove-Item $promptPath, $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
        }

        if ([string]::IsNullOrWhiteSpace($stdoutText)) {
            throw "Gemini CLI nao retornou conteudo util. STDERR:`n$stderrText"
        }

        return @{ Text = $stdoutText.Trim(); ModelUsed = $ModelName; Stderr = $stderrText }
    }
    finally {
        Pop-Location
    }
}

$scopeConfig = Get-ScopeConfig -RequestedScope $Scope
$outputsDir = Join-Path $ProjectRoot 'outputs\hermes'
$chatDir = Join-Path $outputsDir 'chat'
$historyDir = Join-Path $chatDir 'history'
$briefPath = Join-Path $outputsDir 'risk_brief_latest.md'
$csvPath = Join-Path $outputsDir $scopeConfig.Csv
$tactical14dPath = Join-Path $outputsDir 'dados_status_enriquecido_14d_latest.csv'
$tactical14dSummaryMdPath = Join-Path $outputsDir 'dados_status_enriquecido_14d_summary_latest.md'
$tactical14dSummaryJsonPath = Join-Path $outputsDir 'dados_status_enriquecido_14d_summary_latest.json'
$soulPath = Join-Path $HermesWorkspace '.hermes\SOUL.md'
$localHermesPath = Join-Path $ProjectRoot '.hermes.md'

New-Item -ItemType Directory -Path $historyDir -Force | Out-Null

$soulText = Read-OptionalFile -Path $soulPath
$localHermesText = Read-OptionalFile -Path $localHermesPath
$briefText = Read-OptionalFile -Path $briefPath
$csvExcerpt = Get-CsvExcerpt -Path $csvPath
$tactical14dSummaryMd = Read-OptionalFile -Path $tactical14dSummaryMdPath
$tactical14dSummaryJson = Read-OptionalFile -Path $tactical14dSummaryJsonPath
$querySpecificContext = Get-QuerySpecificContext -Query $Query -ScopeCsvPath $csvPath -TacticalCsvPath $tactical14dPath -RequestedScope $Scope
$tactical14dContext = if (-not [string]::IsNullOrWhiteSpace($tactical14dSummaryMd)) {
    $tactical14dSummaryMd
}
else {
    Get-Tactical14dContext -Path $tactical14dPath -RequestedScope $Scope -ScopeCsvPath $csvPath
}

$prompt = @"
Voce esta respondendo como um assistente que recebeu a memoria util do Hermes para o projeto Report Preview.

Objetivo:
- responder a pergunta do usuario com base nos artefatos do projeto;
- manter estilo pt-BR, objetivo e analitico;
- usar leitura gerencial e identificar padroes quando existirem;
- entregar sempre uma previsao operacional para os proximos 7 dias, nunca uma resposta apenas retrospectiva;
- nao inventar rankings, nomes ou causalidade fora do contexto recebido;
- se os artefatos Hermes nao trouxerem referencia suficiente para responder diretamente, usar obrigatoriamente o CSV tatico dos ultimos 14 dias para construir uma analise independente;
- priorizar o resumo tatico 14d pre-calculado quando ele existir, porque ele ja agrega focos territoriais e padroes operacionais do CSV bruto;
- se houver dados uteis no CSV tatico 14d, nunca responder com ausencia total de base;
- quando a resposta vier principalmente do CSV tatico 14d, explicitar que se trata de projecao tatico-operacional para os proximos 7 dias sustentada pelos ultimos 14 dias e que ela nao incorpora necessariamente os artefatos Hermes mais recentes.
- se a pergunta mencionar um bairro, cidade ou localidade especifica, usar obrigatoriamente a secao `ALVO ESPECIFICO DA PERGUNTA` antes de concluir que nao ha referencia;
- se nao houver match direto da localidade no CSV tatico 14d, ainda assim responder com a melhor leitura tatica do escopo e dizer explicitamente apenas que nao houve correspondencia direta da localidade na janela, sem encerrar a resposta nisso.

Regras obrigatorias de previsao:
- toda resposta deve orientar a decisao para os proximos 7 dias;
- usar primeiro o Hermes como base preditiva quando houver ranking, score, driver ou localidade no snapshot atual;
- usar o tatico 14d como sustentacao, aceleracao, concentracao territorial e padrao operacional para a projecao dos proximos 7 dias;
- quando houver localidade especifica no snapshot Hermes, transformar score, driver e contexto tatico em previsao objetiva para os proximos 7 dias;
- quando nao houver localidade especifica no snapshot ou no CSV tatico, ainda assim produzir previsao para os proximos 7 dias com base no melhor contexto territorial do escopo, deixando explicita a limitacao da localidade;
- nunca responder apenas que nao encontrou dados; sempre devolver uma previsao operacional util para os proximos 7 dias.

Formato preferido:
1. Dados ate
2. Fonte
3. Leitura rapida
4. Previsao para os proximos 7 dias
5. Por que importa
6. Proxima acao

MEMORIA GLOBAL HERMES (SOUL):
$soulText

MEMORIA LOCAL DO PROJETO (.hermes.md):
$localHermesText

BRIEF HERMES:
$briefText

EXTRATO CSV DO ESCOPO $($scopeConfig.Label):
$csvExcerpt

ALVO ESPECIFICO DA PERGUNTA:
$querySpecificContext

CONTEXTO TATICO INDEPENDENTE DOS ULTIMOS 14 DIAS:
$tactical14dContext

RESUMO TATICO 14D EM JSON (quando precisar de estrutura):
$tactical14dSummaryJson

PERGUNTA DO USUARIO:
$Query
"@

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$latestAnswerPath = Join-Path $chatDir ('gemini_chat_{0}_latest.md' -f $Scope)
$historyAnswerPath = Join-Path $historyDir ('gemini_chat_{0}_{1}.md' -f $Scope, $timestamp)
$historyPromptPath = Join-Path $historyDir ('gemini_chat_prompt_{0}_{1}.txt' -f $Scope, $timestamp)

$result = Invoke-GeminiText -PromptText $prompt -WorkingDirectory $ProjectRoot -ModelName $GeminiModel

$content = @(
    '# Resposta Gemini com Memoria Hermes',
    '',
    ('Gerado em: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
    ('Escopo: ' + $scopeConfig.Label),
    ('Modelo Gemini CLI: ' + $result.ModelUsed),
    ('Pergunta: ' + $Query),
    ('CSV base: outputs/hermes/' + $scopeConfig.Csv),
    ('CSV tatico 14d: outputs/hermes/dados_status_enriquecido_14d_latest.csv'),
    '',
    $result.Text,
    ''
) -join [Environment]::NewLine

[System.IO.File]::WriteAllText($latestAnswerPath, $content, [System.Text.Encoding]::UTF8)
[System.IO.File]::WriteAllText($historyAnswerPath, $content, [System.Text.Encoding]::UTF8)
[System.IO.File]::WriteAllText($historyPromptPath, $prompt, [System.Text.Encoding]::UTF8)

Write-Host "Resposta salva em: $latestAnswerPath"
Write-Host "Historico da resposta: $historyAnswerPath"
Write-Host "Prompt salvo em: $historyPromptPath"
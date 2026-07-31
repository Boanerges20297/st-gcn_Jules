from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RESUMO_EXECUTIVO_SUPESP_REPORT_PREVIEW.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1769AA")
PALE_BLUE = colors.HexColor("#E8F1F8")
TEAL = colors.HexColor("#087E8B")
PALE_TEAL = colors.HexColor("#E6F4F4")
RED = colors.HexColor("#B42318")
PALE_RED = colors.HexColor("#FDECEC")
GREEN = colors.HexColor("#1F6F50")
PALE_GREEN = colors.HexColor("#E8F5EE")
INK = colors.HexColor("#17202A")
MUTED = colors.HexColor("#52606D")
GRID = colors.HexColor("#AAB7C4")
WHITE = colors.white


PT_WORDS = {
    "acoes": "ações", "agregacao": "agregação", "alem": "além",
    "alteracoes": "alterações", "analise": "análise", "analises": "análises",
    "analitico": "analítico", "analiticos": "analíticos", "aparencia": "aparência",
    "aplicacao": "aplicação", "apreensoes": "apreensões", "area": "área",
    "areas": "áreas", "ate": "até", "atras": "atrás", "atualizacao": "atualização",
    "avaliacao": "avaliação", "avaliavel": "avaliável", "avaliaveis": "avaliáveis",
    "auditavel": "auditável", "auditaveis": "auditáveis", "bairro": "bairro",
    "calibracao": "calibração", "calendario": "calendário", "captura": "captura",
    "codigo": "código", "codigos": "códigos", "codificacao": "codificação",
    "comparacao": "comparação", "compativel": "compatível",
    "compreensiveis": "compreensíveis", "configuracao": "configuração",
    "configuracoes": "configurações", "confianca": "confiança",
    "confirmacoes": "confirmações", "consistencia": "consistência",
    "consolidacao": "consolidação", "construcao": "construção",
    "contencao": "contenção", "cooperacao": "cooperação", "correcao": "correção",
    "correcoes": "correções", "credenciais": "credenciais", "criterios": "critérios",
    "critica": "crítica", "criticas": "críticas", "critico": "crítico",
    "dados": "dados", "decisao": "decisão", "decisoes": "decisões",
    "definicoes": "definições", "dependencia": "dependência",
    "dependencias": "dependências", "diario": "diário", "dicionario": "dicionário",
    "dinamicas": "dinâmicas", "disponivel": "disponível", "dominio": "domínio",
    "divergencias": "divergências",
    "evidencia": "evidência", "evidencias": "evidências", "exclusao": "exclusão",
    "estavel": "estável", "estrategica": "estratégica", "estrategico": "estratégico",
    "exclusoes": "exclusões", "exposicao": "exposição", "exportacao": "exportação",
    "exportacoes": "exportações", "faccao": "facção", "faccoes": "facções",
    "gestao": "gestão", "governanca": "governança",
    "geografica": "geográfica", "geograficas": "geográficas",
    "georreferenciamento": "georreferenciamento", "graficos": "gráficos",
    "ha": "há", "historica": "histórica", "historico": "histórico",
    "horario": "horário", "horarios": "horários", "hipotese": "hipótese",
    "identificacao": "identificação", "identificaveis": "identificáveis",
    "implicacao": "implicação", "incorporacao": "incorporação",
    "inferencia": "inferência", "influencia": "influência", "informacao": "informação",
    "institucional": "institucional", "inteligencia": "inteligência",
    "integracao": "integração", "ja": "já", "limitacao": "limitação",
    "limitacoes": "limitações", "metodo": "método", "metricas": "métricas",
    "mantem": "mantém", "metodologica": "metodológica", "metodologicas": "metodológicas",
    "microterritorio": "microterritório", "microterritorios": "microterritórios",
    "minimo": "mínimo", "minimos": "mínimos", "minimizacao": "minimização", "mitigacao": "mitigação",
    "mitigacoes": "mitigações", "municipio": "município", "municipios": "municípios",
    "necessaria": "necessária", "necessarias": "necessárias",
    "necessario": "necessário", "necessarios": "necessários",
    "ocorrencia": "ocorrência", "ocorrencias": "ocorrências",
    "operacao": "operação", "otimizacao": "otimização", "parametros": "parâmetros",
    "periodo": "período", "periodicidade": "periodicidade",
    "poligono": "polígono", "poligonos": "polígonos", "pragmatica": "pragmática",
    "precisao": "precisão", "predicao": "predição", "previsao": "previsão",
    "preferencialmente": "preferencialmente", "provavel": "provável", "provaveis": "prováveis",
    "protecao": "proteção", "propria": "própria", "proprias": "próprias",
    "proximo": "próximo", "proxima": "próxima", "pseudonimo": "pseudônimo",
    "publica": "pública", "publicacao": "publicação",
    "recalibracao": "recalibração", "recomendacao": "recomendação",
    "referencia": "referência", "referencias": "referências", "regiao": "região",
    "regioes": "regiões", "relatorio": "relatório", "relatorios": "relatórios",
    "reproducao": "reprodução", "reprodutibilidade": "reprodutibilidade",
    "restricao": "restrição", "revisao": "revisão", "revisoes": "revisões",
    "rapido": "rápido", "responsavel": "responsável", "rotulos": "rótulos",
    "segregacao": "segregação", "seguranca": "segurança",
    "semantica": "semântica", "sensiveis": "sensíveis",
    "separacao": "separação", "serie": "série", "situacao": "situação",
    "sobreposicao": "sobreposição", "tecnica": "técnica", "tecnico": "técnico",
    "tecnicos": "técnicos", "territorio": "território", "territorios": "territórios",
    "tatico": "tático", "tatica": "tática", "tipologica": "tipológica",
    "supervisao": "supervisão", "transferencia": "transferência",
    "usuarios": "usuários", "validacao": "validação",
    "variacao": "variação", "veiculos": "veículos", "verificacao": "verificação",
    "versao": "versão", "vigencia": "vigência", "visao": "visão",
}


def pt(text):
    phrases = {
        "O limite atual e de dados": "O limite atual é de dados",
        "O modelo padrao e": "O modelo padrão é",
        "esta implementada": "está implementada",
        "esta na": "está na",
        "tecnologia: e a falta": "tecnologia: é a falta",
        "nao e": "não é",
        "nao esta": "não está",
        "Nao ": "Não ",
        "nao ": "não ",
        "padrao": "padrão",
    }
    for source, target in phrases.items():
        text = text.replace(source, target)

    def replace(match):
        source = match.group(0)
        target = PT_WORDS[source.lower()]
        if source.isupper():
            return target.upper()
        if source[:1].isupper():
            return target[:1].upper() + target[1:]
        return target

    pattern = r"\b(" + "|".join(sorted(map(re.escape, PT_WORDS), key=len, reverse=True)) + r")\b"
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


def P(text, style):
    return Paragraph(pt(text), style)


def page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.2)
    canvas.line(1.25 * cm, 28.82 * cm, 19.75 * cm, 28.82 * cm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(1.25 * cm, 0.72 * cm, pt("Report Preview | Cooperacao tecnica e acesso a dados SUPESP"))
    canvas.drawRightString(19.75 * cm, 0.72 * cm, f"{doc.page}")
    canvas.restoreState()


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        name="CoverTitle", parent=s["Title"], fontName="Helvetica-Bold",
        fontSize=27, leading=31, alignment=TA_LEFT, textColor=NAVY, spaceAfter=10,
    ))
    s.add(ParagraphStyle(
        name="Subtitle", parent=s["BodyText"], fontName="Helvetica-Bold",
        fontSize=13.5, leading=17, textColor=BLUE, spaceAfter=12,
    ))
    s.add(ParagraphStyle(
        name="H1", parent=s["Heading1"], fontName="Helvetica-Bold",
        fontSize=17.5, leading=21, textColor=NAVY, spaceBefore=2, spaceAfter=8,
    ))
    s.add(ParagraphStyle(
        name="H2", parent=s["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=5, spaceAfter=4,
    ))
    s.add(ParagraphStyle(
        name="Body", parent=s["BodyText"], fontSize=9.8, leading=13.2,
        textColor=INK, spaceAfter=5,
    ))
    s.add(ParagraphStyle(
        name="BodyBold", parent=s["BodyText"], fontName="Helvetica-Bold",
        fontSize=10.0, leading=13.5, textColor=INK, spaceAfter=5,
    ))
    s.add(ParagraphStyle(
        name="Small", parent=s["BodyText"], fontSize=8.2, leading=10.8,
        textColor=MUTED, spaceAfter=3,
    ))
    s.add(ParagraphStyle(
        name="Cell", parent=s["BodyText"], fontSize=8.15, leading=10.4,
        textColor=INK,
    ))
    s.add(ParagraphStyle(
        name="CellBold", parent=s["BodyText"], fontName="Helvetica-Bold",
        fontSize=8.15, leading=10.4, textColor=NAVY,
    ))
    s.add(ParagraphStyle(
        name="CellHead", parent=s["BodyText"], fontName="Helvetica-Bold",
        fontSize=8.15, leading=10.2, textColor=WHITE, alignment=TA_CENTER,
    ))
    s.add(ParagraphStyle(
        name="Callout", parent=s["BodyText"], fontName="Helvetica-Bold",
        fontSize=10.8, leading=14.2, textColor=NAVY, spaceAfter=0,
    ))
    return s


def callout(text, st, tone="blue"):
    palettes = {
        "blue": (PALE_BLUE, BLUE),
        "red": (PALE_RED, RED),
        "green": (PALE_GREEN, GREEN),
        "teal": (PALE_TEAL, TEAL),
    }
    background, border = palettes[tone]
    table = Table([[P(text, st["Callout"])]], colWidths=[18.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 1.0, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def table(rows, st, widths, header=True, first_col=True, row_colors=None):
    data = []
    for row_index, row in enumerate(rows):
        rendered = []
        for col_index, value in enumerate(row):
            if header and row_index == 0:
                style = st["CellHead"]
            elif first_col and col_index == 0:
                style = st["CellBold"]
            else:
                style = st["Cell"]
            rendered.append(P(str(value), style))
        data.append(rendered)

    result = Table(
        data,
        colWidths=[width * cm for width in widths],
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )
    commands = [
        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
        body_start = 1
    else:
        body_start = 0
    if first_col:
        commands.append(("BACKGROUND", (0, body_start), (0, -1), PALE_BLUE))
    for row_index, color in row_colors or []:
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), color))
    result.setStyle(TableStyle(commands))
    return result


def section(title, body, st):
    return KeepTogether([
        P(title, st["H2"]),
        P(body, st["Body"]),
    ])


def build():
    st = styles()
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=1.25 * cm,
        leftMargin=1.25 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.25 * cm,
        title="Cooperacao tecnica e acesso a dados SUPESP",
        author="Report Preview",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=page)])
    story = []

    story += [
        Spacer(1, 0.35 * cm),
        P("Report Preview", st["CoverTitle"]),
        P("Proposta de cooperacao tecnica e acesso a dados da SUPESP", st["Subtitle"]),
        callout(
            "DECISAO SOLICITADA: autorizar um fluxo institucional de dados oficiais, "
            "com atualizacao recorrente e focal tecnico, para validacao prospectiva de "
            "inteligencia territorial de CVLI em horizonte de 30 dias.",
            st,
            "blue",
        ),
        Spacer(1, 0.25 * cm),
        P("Por que este acesso e necessario", st["H1"]),
        P(
            "O projeto ja executa modelos, integra escalas territoriais e produz artefatos "
            "auditaveis. O limite atual nao e a ausencia de tecnologia: e a falta de um "
            "fluxo oficial que assegure atualidade, completude, estabilidade semantica e "
            "validacao institucional dos resultados.",
            st["Body"],
        ),
        table([
            ["Capacidade existente", "O que ja funciona", "O que os dados SUPESP destravam"],
            ["ST-GAT v5", "Ranking estrategico de Fortaleza, RMF e Interior.", "Treino e validacao com eventos oficiais, cortes temporais e revisoes."],
            ["Perfis temporais", "Dia e janela de 5 horas de maior pressao.", "Estabilidade horaria por territorio e atualizacao recorrente."],
            ["GA espacial", "Otimizacao de captura, area e sobreposicao.", "Sinal local suficiente para microterritorios defensaveis."],
            ["ST-GCN", "Focos de rua, propagacao espacial e rotas provaveis.", "Calibracao microterritorial e avaliacao de utilidade em campo."],
            ["Hermes", "Briefings, snapshots e historico de respostas.", "Produtos gerenciais com fonte oficial e data de corte verificavel."],
        ], st, [3.0, 6.2, 8.8]),
        PageBreak(),
    ]

    story += [
        P("1. Capacidade instalada", st["H1"]),
        P(
            "A estrutura ja cobre o ciclo completo: ingestao, enriquecimento, construcao de "
            "grafos, inferencia, explicabilidade, exportacao, monitoramento e consumo gerencial.",
            st["Body"],
        ),
        table([
            ["Escala", "Componente", "Entrega"],
            ["Estrategica", "ST-GAT v5", "Prioriza bairros e municipios a partir de dependencias espaciais e temporais."],
            ["Regional", "3 especialistas", "Fortaleza, Regiao Metropolitana e Interior com processamento separado."],
            ["Temporal", "Perfis preditivos", "Calcula dia da semana e faixa critica de 5 horas no horizonte de 30 dias."],
            ["Tatica", "GA multiobjetivo", "Compara captura futura, area coberta, sobreposicao e legibilidade operacional."],
            ["Micro", "ST-GCN rua/foco", "Propaga sinal entre focos vizinhos e combina risco da area, historico local e densidade viaria."],
            ["Resposta", "ST-GCN rotas", "Estima eixos provaveis, malha de alcance e perimetro de contencao a partir de um evento."],
            ["Territorial", "Micronodos/ORCRIMS", "Integra poligonos, tensao territorial, faccoes e camadas oficiais ou de referencia."],
            ["Gestao", "Hermes", "Gera ranking, drivers, recortes regionais, briefings e historico auditavel."],
            ["Governanca", "Validacao/monitoramento", "Registra metricas, snapshots, versao de modelo e comportamento recente."],
        ], st, [2.5, 4.1, 11.4]),
        Spacer(1, 0.22 * cm),
        P("Fluxo ja implementado", st["H2"]),
        table([
            ["1", "Dados brutos e territoriais"],
            ["2", "Padronizacao, georreferenciamento e construcao dos grafos"],
            ["3", "ST-GAT v5 para prioridade estrategica em 30 dias"],
            ["4", "GA e ST-GCN para refinamento tatico e microterritorial"],
            ["5", "Hermes, dashboard e exportacoes para leitura gerencial"],
            ["6", "Validacao prospectiva, monitoramento e recalibracao"],
        ], st, [1.0, 17.0], header=False),
        PageBreak(),
    ]

    story += [
        P("2. O limite atual e de dados", st["H1"]),
        P(
            "A base local comprova capacidade de processamento em escala, mas tambem revela "
            "lacunas que afetam a precisao territorial e a validade institucional.",
            st["Body"],
        ),
        table([
            ["Indicador observado", "Situacao atual", "Implicacao"],
            ["Registros processados", "143.617 ocorrencias", "Volume suficiente para pipeline, testes e operacao experimental."],
            ["Eventos CVLI", "12.633 eventos em 182 municipios", "Cobertura estadual relevante para modelos regionais."],
            ["Data, hora e coordenadas", "Preenchimento integral nos CVLI locais", "Sustenta series temporais e georreferenciamento inicial."],
            ["Bairro nos CVLI", "53,6% preenchido", "Limita a analise intramunicipal e aumenta dependencia de inferencia geografica."],
            ["Natureza detalhada", "48,4% preenchido", "Reduz separacao entre dinamicas criminais distintas."],
            ["Area territorial de faccao", "0% no campo da base principal", "Impede usar essa dimensao como dado historico versionado e rastreavel."],
        ], st, [4.0, 4.2, 9.8], row_colors=[(4, PALE_RED), (5, PALE_RED), (6, PALE_RED)]),
        Spacer(1, 0.2 * cm),
        P("Por que a granularidade muda o resultado", st["H2"]),
        P(
            "O bairro e hoje a unidade estrategica mais defensavel. Experimentos com "
            "microterritorios produziram resultados distintos conforme o sinal local e a "
            "restricao de area. Sem dado oficial granular, uma geometria limpa pode parecer "
            "precisa sem melhorar a captura futura. O acesso solicitado permite testar essa "
            "hipotese corretamente e interromper abordagens que nao agreguem valor.",
            st["Body"],
        ),
        table([
            ["Evidencia experimental", "Resultado", "Leitura pragmatica"],
            ["Ranking por bairro", "Ate 89,04% de captura em uma configuracao avaliada.", "Baseline estrategico forte; permanece como referencia."],
            ["GA espacial amplo", "Ate 83,6% em 109 janelas, com area operacional elevada.", "Captura melhora, mas custo territorial precisa ser limitado."],
            ["Colmeia local", "Entre 13,8% e 34,37% em configuracoes recentes.", "Sem sinal local robusto, o hexagono nao justifica uso operacional."],
        ], st, [4.0, 4.8, 9.2]),
        PageBreak(),
    ]

    story += [
        P("3. Dados solicitados e finalidade", st["H1"]),
        P(
            "O pedido segue minimizacao: somente campos necessarios ao objetivo analitico, "
            "preferencialmente sem nomes, documentos pessoais ou narrativas identificaveis.",
            st["Body"],
        ),
        table([
            ["Prioridade", "Conjunto minimo", "Campos essenciais", "Uso direto"],
            ["1", "CVLI historico por evento", "ID pseudonimo, data/hora, municipio, bairro, AIS/RISP, latitude/longitude, natureza, meio, mortes, status.", "Treino ST-GAT, perfis temporais, validacao e correcao de rotulos."],
            ["2", "Atualizacao incremental", "Novos eventos, alteracoes, exclusoes, data de consolidacao e data de corte.", "Ranking de 30 dias atualizado, controle de atraso e avaliacao prospectiva."],
            ["3", "Malhas territoriais oficiais", "Municipio, bairro, AIS, RISP, micronodos e vigencia de cada poligono.", "Grafo espacial, joins corretos, comparacao entre escalas e mapas auditaveis."],
            ["4", "Atividade operacional agregada", "Acoes, apreensoes, armas, drogas, veiculos e mandados por territorio/data.", "Distinguir pressao criminal, resposta estatal e mudanca de exposicao."],
            ["5", "Inteligencia territorial", "Areas de influencia, disputa, mudanca de dominio, fonte, confianca e vigencia.", "Calibrar tensao territorial sem transformar dado desatualizado em verdade fixa."],
            ["6", "Dicionario e qualidade", "Definicoes, codigos, regras de consolidacao, cobertura, atrasos e revisoes.", "Reprodutibilidade, comparabilidade e governanca do modelo."],
        ], st, [1.2, 3.5, 7.3, 6.0]),
        Spacer(1, 0.22 * cm),
        P("Formato e periodicidade", st["H2"]),
        table([
            ["Carga historica", "Serie integral disponivel, preferencialmente cinco anos ou mais."],
            ["Incremental", "Diario ou semanal, sempre com identificador estavel e registros corrigidos."],
            ["Formato", "CSV/Parquet e GeoJSON/KML, com dicionario de dados e codificacao definida."],
            ["Canal", "Transferencia institucional controlada, com registro de recebimento e integridade."],
            ["Focal tecnico", "Um responsavel por semantica, mudancas de esquema e calendario de atualizacao."],
        ], st, [4.0, 14.0], header=False),
        PageBreak(),
    ]

    story += [
        P("4. O que a SUPESP recebe", st["H1"]),
        P(
            "A cooperacao produz entregas recorrentes para gestao, avaliacao e planejamento, "
            "com separacao clara entre previsao, evidencia historica e decisao humana.",
            st["Body"],
        ),
        table([
            ["Entrega", "Conteudo", "Valor para a SUPESP"],
            ["Boletim de 30 dias", "Top territorios ST-GAT, variacao, confianca, drivers e horario de pico.", "Prioridade comparavel entre ciclos, com data de corte explicita."],
            ["Mapa estrategico", "Bairros e municipios de maior risco estimado.", "Visao estadual e regional sem falsa precisao pontual."],
            ["Camada tatico-experimental", "Microterritorios GA e focos ST-GCN com area, captura e limitacoes.", "Teste controlado antes de qualquer incorporacao operacional."],
            ["Avaliacao prospectiva", "P@10, P@20, recall, cobertura, captura por area e estabilidade regional.", "Saber onde o modelo ajuda, onde falha e quando deve ser recalibrado."],
            ["Briefing Hermes", "Leitura executiva, fonte, justificativa e proxima verificacao.", "Consumo gerencial rapido com rastreabilidade."],
            ["Relatorio de qualidade", "Completude, atraso, duplicidade, mudanca de esquema e impacto.", "Melhoria do ativo de dados, alem do modelo."],
            ["Pacote de auditoria", "Versao da base, corte temporal, modelo, parametros e artefatos publicados.", "Reproducao de cada resultado apresentado."],
        ], st, [3.9, 7.0, 7.1]),
        Spacer(1, 0.25 * cm),
        P("Criterios de sucesso do piloto", st["H2"]),
        table([
            ["Validade", "Desempenho prospectivo superior ou complementar aos baselines por regiao."],
            ["Utilidade", "Alertas compreensiveis e avaliaveis pela equipe tecnica da SUPESP."],
            ["Territorio", "Ganho de captura compativel com area operacional e sem sobreposicao excessiva."],
            ["Tempo", "Estabilidade dos horarios de pico com suporte amostral informado."],
            ["Governanca", "Todo resultado vinculado a dado, versao, corte e modelo."],
            ["Decisao", "Nenhuma recomendacao automatica de emprego policial; apoio analitico documentado."],
        ], st, [3.2, 14.8], header=False),
        PageBreak(),
    ]

    story += [
        P("5. Governanca e protecao", st["H1"]),
        P(
            "O uso proposto preserva controle institucional, minimiza dados pessoais e "
            "mantem a decisao operacional sob responsabilidade humana.",
            st["Body"],
        ),
        table([
            ["Controle", "Compromisso operacional"],
            ["Finalidade definida", "Uso restrito a pesquisa aplicada, validacao e apoio analitico de CVLI."],
            ["Minimizacao", "Recebimento apenas dos campos necessarios; exclusao de identificadores pessoais diretos."],
            ["Acesso restrito", "Usuarios autorizados, credenciais individuais e trilha de acesso."],
            ["Ambiente controlado", "Processamento local ou em infraestrutura acordada, sem exposicao publica da base bruta."],
            ["Versionamento", "Registro da carga, esquema, data de corte, revisoes e artefatos derivados."],
            ["Retencao e descarte", "Prazo acordado, copia controlada e eliminacao verificavel ao encerramento."],
            ["Publicacao", "Somente resultados agregados e previamente enquadrados pela cooperacao."],
            ["Supervisao humana", "Modelos indicam prioridade para avaliacao; nao determinam emprego operacional."],
            ["Revisao conjunta", "Resultados sensiveis e mudancas metodologicas avaliados com focal tecnico da SUPESP."],
        ], st, [4.2, 13.8]),
        Spacer(1, 0.25 * cm),
        P("Riscos e mitigacoes", st["H2"]),
        table([
            ["Risco", "Mitigacao"],
            ["Falsa precisao territorial", "Bairro como baseline; microterritorio somente com evidencia superior e area controlada."],
            ["Defasagem ou revisao tardia", "Carga incremental com status, consolidacao, exclusoes e data de corte."],
            ["Mudanca de padrao regional", "Metricas separadas para Fortaleza, RMF e Interior."],
            ["Vies de atividade policial", "Incluir atividade operacional agregada e separar ocorrencia de resposta estatal."],
            ["Uso indevido do score", "Linguagem probabilistica, confianca, drivers, limitacoes e decisao humana."],
            ["Vazamento de informacao", "Minimizacao, segregacao de acesso, auditoria e publicacao agregada."],
        ], st, [5.0, 13.0]),
        PageBreak(),
    ]

    story += [
        P("6. Piloto conjunto em 30 dias", st["H1"]),
        table([
            ["Periodo", "Atividade conjunta", "Entrega verificavel"],
            ["Dias 1 a 5", "Formalizar finalidade, focal tecnico, dicionario, canal e recorte inicial.", "Termo operacional, esquema validado e checklist de seguranca."],
            ["Dias 6 a 10", "Receber carga historica, medir qualidade e reconciliar territorios.", "Relatorio de completude, cobertura, duplicidade e consistencia geografica."],
            ["Dias 11 a 17", "Reconstruir grafos e executar ST-GAT v5 e baselines.", "Ranking inicial por regiao e protocolo de validacao prospectiva."],
            ["Dias 18 a 23", "Calcular horarios de pico e testar GA/ST-GCN em recorte controlado.", "Camadas experimentais com captura, area, suporte e limitacoes."],
            ["Dias 24 a 27", "Revisar resultados com analistas da SUPESP e incorporar retorno.", "Registro de confirmacoes, divergencias e ajustes."],
            ["Dias 28 a 30", "Consolidar desempenho, governanca e continuidade.", "Boletim executivo, pacote de auditoria e decisao sobre o proximo ciclo."],
        ], st, [2.5, 8.4, 7.1]),
        Spacer(1, 0.25 * cm),
        P("Decisoes necessarias nesta etapa", st["H2"]),
        table([
            ["1", "Autorizar o compartilhamento institucional do conjunto minimo descrito."],
            ["2", "Designar focal tecnico para dados, semantica e validacao."],
            ["3", "Definir recorte inicial: Fortaleza, RMF ou cobertura estadual."],
            ["4", "Aprovar periodicidade e canal seguro de atualizacao."],
            ["5", "Iniciar o piloto de 30 dias com criterios de sucesso acordados."],
        ], st, [1.0, 17.0], header=False),
        Spacer(1, 0.3 * cm),
        P(
            "<b>Resultado esperado:</b> ao final do primeiro ciclo, a SUPESP tera evidencias "
            "para decidir se a capacidade agrega valor operacional, em quais regioes, com "
            "qual granularidade e sob quais limites.",
            st["BodyBold"],
        ),
        P(
            "Base desta proposta: codigo, APIs, pipeline de dados, artefatos Hermes, "
            "monitoramento, experimentos espaciais e perfil da base local verificados em 27/07/2026.",
            st["Small"],
        ),
    ]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "RESUMO_EXECUTIVO_ESTRATEGICO_SUPESP.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F9E")
TEAL = colors.HexColor("#287D78")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#5C6975")


def P(text, style):
    return Paragraph(text, style)


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1)
    canvas.line(1.4 * cm, 28.78 * cm, 19.6 * cm, 28.78 * cm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.4)
    canvas.drawString(
        1.4 * cm,
        0.7 * cm,
        "Report Preview | Resumo executivo estratégico para a SUPESP",
    )
    canvas.drawRightString(19.6 * cm, 0.7 * cm, str(doc.page))
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="CoverTitle", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=26, leading=30, alignment=TA_LEFT, textColor=NAVY, spaceAfter=9,
    ))
    base.add(ParagraphStyle(
        name="Subtitle", parent=base["BodyText"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=16, textColor=BLUE, spaceAfter=13,
    ))
    base.add(ParagraphStyle(
        name="H1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=16.5, leading=20, textColor=NAVY, spaceBefore=3, spaceAfter=7,
    ))
    base.add(ParagraphStyle(
        name="H2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.8, leading=13.5, textColor=TEAL, spaceBefore=6, spaceAfter=3,
    ))
    base.add(ParagraphStyle(
        name="Lead", parent=base["BodyText"], fontSize=10.8, leading=15,
        textColor=INK, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="Body", parent=base["BodyText"], fontSize=9.4, leading=13.2,
        textColor=INK, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="Small", parent=base["BodyText"], fontSize=7.8, leading=10.3,
        textColor=MUTED, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="ExecBullet", parent=base["BodyText"], fontSize=9.1, leading=12.6,
        textColor=INK, leftIndent=2,
    ))
    return base


def bullets(items, st):
    return ListFlowable(
        [ListItem(P(item, st["ExecBullet"]), leftIndent=12) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=17,
        bulletFontName="Helvetica",
        bulletFontSize=6.5,
        bulletColor=BLUE,
        spaceAfter=6,
    )


def build():
    st = styles()
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.32 * cm,
        bottomMargin=1.3 * cm,
        title="Resumo executivo estratégico - Report Preview",
        author="Report Preview",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=draw_page)])
    story = []

    story += [
        Spacer(1, 0.35 * cm),
        P("Report Preview", st["CoverTitle"]),
        P("Inteligência territorial e prospectiva de CVLI", st["Subtitle"]),
        P(
            "O Report Preview é uma plataforma completa de engenharia de dados, modelagem "
            "espaço-temporal e apoio à decisão para Crimes Violentos Letais Intencionais. "
            "Ela transforma ocorrências, contexto territorial, calendário, clima, atividade "
            "operacional e relações entre localidades em prioridades analíticas para um "
            "horizonte de 30 dias.",
            st["Lead"],
        ),
        P("Capacidade central", st["H1"]),
        P(
            "O produto não se limita a um mapa ou a um ranking. A plataforma mantém um ciclo "
            "integrado de ingestão, tratamento, construção de grafos, inferência regional, "
            "explicabilidade, simulação de eventos, validação prospectiva, publicação de "
            "artefatos e monitoramento de saúde.",
            st["Body"],
        ),
        bullets([
            "<b>Priorização estratégica:</b> bairros de Fortaleza e municípios da Região Metropolitana e do Interior.",
            "<b>Leitura temporal:</b> tendência recente, momentum, períodos de calmaria, dia da semana e faixa horária de maior pressão.",
            "<b>Leitura territorial:</b> proximidade geográfica, conflito entre grupos, micronodos, AIS/RISP e camadas ORCRIMS.",
            "<b>Refinamento tático:</b> focos de rua, microterritórios experimentais e análise de rotas prováveis.",
            "<b>Contexto operacional:</b> eventos críticos, ações de supressão, apreensões e sinais de intencionalidade.",
            "<b>Gestão:</b> confiança, expressividade no ranking, fatores explicativos, briefings e trilha auditável.",
        ], st),
        P("Posicionamento dos modelos", st["H2"]),
        P(
            "O ST-GAT v5 é o modelo padrão da experiência operacional. Poisson funciona como "
            "baseline, alternativa explicável e mecanismo de contingência. ST-GCN, LightGBM e "
            "outras arquiteturas permanecem como componentes especializados ou trilhas de "
            "comparação, sem serem apresentadas como equivalentes ao modelo principal.",
            st["Body"],
        ),
        P(
            "A decisão final permanece humana. Os resultados indicam prioridades para análise "
            "e validação, não determinações automáticas de emprego operacional.",
            st["Small"],
        ),
        PageBreak(),
    ]

    story += [
        P("Engenharia de dados e representação territorial", st["H1"]),
        P(
            "O processamento converte registros heterogêneos em séries diárias por território "
            "e em duas estruturas de grafo. Fortaleza é representada por bairros; RMF e Interior "
            "são organizados por municípios. A seleção de nós utiliza corte temporal para evitar "
            "que informação futura contamine a formação da malha.",
            st["Lead"],
        ),
        P("Conjunto de sinais processados", st["H2"]),
        P(
            "A matriz-base possui 37 canais. O pipeline neural amplia a leitura com descritores "
            "de momentum e admite entrada de até 41 canais. Os sinais são organizados em grupos:",
            st["Body"],
        ),
        bullets([
            "<b>Criminalidade:</b> número de mortes por CVLI, crimes de veículo, ocorrências com arma de fogo, disparos, invasões e série estadual agregada.",
            "<b>Recência:</b> soma móvel de sete dias, janelas de 14, 30 e 60 dias, diferenças entre períodos e sequência de dias sem CVLI.",
            "<b>Calendário:</b> sete dias da semana, doze meses, fim de semana, feriados e dias historicamente sensíveis para CVP.",
            "<b>Ambiente:</b> precipitação diária e indicador de chuva significativa.",
            "<b>Território:</b> facção predominante, grau de domínio, disputa e índice de tensão.",
            "<b>Resposta estatal:</b> apreensões de armas, drogas e veículos, mandados e outros registros de atividade da tropa.",
            "<b>Ruptura:</b> eventos com múltiplas mortes e sinais de intencionalidade capazes de elevar o pulso local.",
        ], st),
        P("Dois grafos complementares", st["H2"]),
        P(
            "O primeiro grafo representa proximidade tática: localidades próximas recebem peso "
            "inversamente proporcional à distância, com reforço quando há rivalidade e atividade "
            "operacional associada. O segundo representa conflito territorial entre áreas de "
            "grupos distintos. O modelo aprende sobre os dois grafos para diferenciar vizinhança "
            "geográfica de relação de conflito.",
            st["Body"],
        ),
        P("Qualidade e enriquecimento", st["H2"]),
        P(
            "O pipeline normaliza nomes, resolve homônimos entre bairro e município, corrige "
            "localização por coordenadas e polígonos, consolida múltiplas mortes no mesmo evento, "
            "atualiza cache de ruas e preserva metadados regionais. Regras heurísticas separam "
            "sinais táticos de casos com dinâmica distinta, sempre sujeitas a revisão com dados "
            "oficiais e conhecimento de domínio.",
            st["Body"],
        ),
        PageBreak(),
    ]

    story += [
        P("Ecossistema de modelos e orquestração", st["H1"]),
        P(
            "O orquestrador carrega dados, modelos e parâmetros por região, unifica os escores, "
            "aplica contexto recente, registra componentes explicativos e produz artefatos para "
            "consumo gerencial. Janelas históricas podem variar por especialista sem alterar o "
            "contrato de projeção de 30 dias.",
            st["Lead"],
        ),
        P("ST-GAT v5 - modelo padrão", st["H2"]),
        P(
            "O DeepSTGAT_v5 possui três blocos neurais de 48, 96 e 96 unidades. Cada bloco combina "
            "convolução temporal, atenção aprendida por aresta, atenção temporal multi-head, "
            "normalização e conexão residual. A atenção por aresta permite que a influência entre "
            "dois territórios seja aprendida separadamente nos grafos geográfico e de conflito.",
            st["Body"],
        ),
        P(
            "Há especialistas próprios para Fortaleza, RMF e Interior. Na inferência, o sinal "
            "neural é combinado com atividade recente e suporte territorial. Tensões históricas "
            "não promovem isoladamente uma área fria; a plataforma exige evidência atual ou "
            "continuidade territorial suficiente.",
            st["Body"],
        ),
        P("Modelos complementares disponíveis", st["H2"]),
        bullets([
            "<b>Poisson regional:</b> 23 variáveis de defasagem, somas, médias, momentum, sazonalidade e tensão; baixo custo e alta explicabilidade.",
            "<b>LightGBM challenger:</b> refinamento de Fortaleza com peso dinâmico limitado e arbitragem por desempenho recente.",
            "<b>DeepSTGAT 32/64/80:</b> variantes de capacidade para benchmark, estabilidade e estudos de arquitetura.",
            "<b>ShallowGAT:</b> modelo residual mais curto, voltado à extração de sinal tático.",
            "<b>PureSTGCN:</b> convolução temporal e grafo relacional sem atenção, útil para comparação arquitetural.",
            "<b>FortalezaHeteroSTGAT:</b> separa sinais dinâmicos dos canais contextuais e aprende uma fusão entre os dois ramos.",
            "<b>Suíte estocástica:</b> baselines zero, lag, média móvel, regressão Poisson, classificadores, hurdle Poisson e modelos deep.",
        ], st),
        P("Orquestração adaptativa", st["H2"]),
        P(
            "O sistema mantém cache de risco, fallback controlado, simulação de intensificação ou "
            "supressão, ajuste determinístico da janela conforme P@10 e persistência do estado de "
            "calibração. O objetivo é degradar de forma previsível, sem ocultar qual modelo ou "
            "janela produziu cada resultado.",
            st["Body"],
        ),
        PageBreak(),
    ]

    story += [
        P("Capacidades analíticas e operacionais", st["H1"]),
        P(
            "Sobre o núcleo preditivo, a aplicação organiza diferentes escalas de decisão e "
            "mantém o vínculo entre score, evidência, território e artefato publicado.",
            st["Lead"],
        ),
        P("Risco, tempo e explicabilidade", st["H2"]),
        bullets([
            "Rankings de bairros e municípios por região, com nível de risco, confiança, percentil e separação frente aos pares.",
            "Drivers ordenados: sinal do modelo, tensão territorial, atividade recente, vizinhança e suporte histórico.",
            "Perfis horários calculados a partir de CVLI real, com dia da semana, janela crítica de cinco horas, participação e tamanho da amostra.",
            "Leitura de momentum em 7, 14 e 30 dias, penalidade por calmaria e suporte territorial recente.",
            "Explicações gerenciais e acadêmicas com fatores, ressalvas, variáveis e histórico de geração.",
        ], st),
        P("Território e resposta", st["H2"]),
        bullets([
            "Polígonos de bairros, municípios, AIS/RISP, facções, micronodos e camadas ORCRIMS com vigência e fallback controlado.",
            "Micronodos visíveis por Fortaleza, RMF e Interior, enriquecidos com risco, facção e perfil temporal.",
            "Focos de rua avaliados por risco da área-mãe, histórico local, densidade viária e propagação entre vizinhos.",
            "Motor ST-GCN para malha viária, perímetro de contenção e eixos prováveis de deslocamento a partir de um ponto.",
            "Eventos exógenos classificados como conflito, atividade policial qualificada ou evento administrativo, com efeitos distintos sobre risco e alívio.",
            "Simulação de cenários e sincronização de eventos por fluxo estruturado, inclusive planilhas institucionais.",
        ], st),
        P("Produtos e observabilidade", st["H2"]),
        bullets([
            "Hermes gera briefings, snapshots JSON/CSV/Markdown, recortes regionais, ruas críticas e histórico por execução.",
            "Exportação estática publica JSON e GeoJSON para painéis desacoplados do servidor de inferência.",
            "Monitor de eficiência mede P@10, P@20, recall e cobertura por região contra eventos observados.",
            "Monitor de saúde acompanha disponibilidade, latência, alertas, confiança e tendência das métricas.",
            "Sincronização remota usa impressão digital de artefatos para evitar publicação redundante.",
        ], st),
        PageBreak(),
    ]

    story += [
        P("Necessidades de dados e expansão", st["H1"]),
        P(
            "A plataforma já executa o ciclo técnico, mas sua maturidade institucional depende "
            "de dados oficiais contínuos, documentados e revisáveis. O valor adicional não está "
            "apenas em aumentar o volume: está em conhecer a semântica, a vigência e o atraso de "
            "cada campo.",
            st["Lead"],
        ),
        P("Dados prioritários da SUPESP", st["H2"]),
        bullets([
            "<b>CVLI por evento:</b> identificador pseudônimo, data, hora, município, bairro, AIS/RISP, coordenadas, natureza, meio, mortes e status de consolidação.",
            "<b>Fluxo incremental:</b> inclusões, correções, exclusões, data de corte e identificador estável para conciliação.",
            "<b>Territórios oficiais:</b> polígonos e vigências de municípios, bairros, AIS, RISP, micronodos e áreas de inteligência.",
            "<b>Atividade operacional agregada:</b> apreensões, mandados, armas, drogas, veículos e ações por território e período.",
            "<b>Metadados:</b> dicionário, códigos, regras de consolidação, cobertura, mudança de esquema e atraso esperado.",
        ], st),
        P("Funcionalidades viabilizadas por esse acesso", st["H2"]),
        bullets([
            "Validação prospectiva recorrente do ST-GAT v5 e dos baselines em cada região.",
            "Calibração de incerteza e confiança com suporte amostral conhecido.",
            "Microterritórios com sinais próprios, em vez de herdar apenas o score do bairro.",
            "Avaliação de efeitos de resposta estatal sem confundir policiamento com pressão criminal.",
            "Detecção de mudança de regime, deriva territorial e degradação de modelo.",
            "Model cards regionais, trilha de promoção e comparação champion/challenger.",
            "Integração institucional com atualização, auditoria e publicação agregada.",
        ], st),
        P("Perspectiva de crescimento", st["H2"]),
        P(
            "A arquitetura permite expandir por território, natureza criminal e horizonte sem "
            "substituir todo o sistema. O crescimento esperado é a evolução de um ranking "
            "prospectivo para uma infraestrutura de inteligência: cenários, alertas de mudança, "
            "avaliação de intervenção, modelos especializados e acompanhamento longitudinal. "
            "Cada expansão deve permanecer condicionada a validação independente e utilidade "
            "operacional mensurável.",
            st["Body"],
        ),
        PageBreak(),
    ]

    story += [
        P("Linha de pesquisa e maturidade metodológica", st["H1"]),
        P(
            "O Report Preview sustenta uma agenda de pesquisa aplicada em previsão de violência "
            "letal sobre dados esparsos, sobredispersos e territorialmente heterogêneos. A "
            "pesquisa compara modelos complexos com baselines simples e registra tanto ganhos "
            "quanto resultados negativos.",
            st["Lead"],
        ),
        P("Eixos de pesquisa", st["H2"]),
        bullets([
            "<b>Aprendizado em grafos:</b> atenção por aresta, múltiplas relações territoriais e especialização regional.",
            "<b>Dados de contagem:</b> Poisson, binomial negativa, hurdle e modelos capazes de representar sobredispersão e excesso de zeros.",
            "<b>Recência e memória:</b> momentum, decaimento temporal, recorrência e períodos de calmaria.",
            "<b>Otimização territorial:</b> captura futura, área coberta, sobreposição, contiguidade e legibilidade.",
            "<b>Explicabilidade e incerteza:</b> confiança, expressividade, fatores associados e limites de decisão.",
            "<b>Interação humano-modelo:</b> retorno estruturado de analistas e validação de campo.",
        ], st),
        P("Resultados que orientam o produto", st["H2"]),
        P(
            "Os experimentos confirmam o bairro como unidade estratégica mais defensável. Filtros "
            "de recorrência e decaimento melhoraram a captura em alguns cenários. Métodos greedy "
            "e GA alcançaram maior captura espacial quando aceitaram áreas amplas, mostrando que "
            "o problema é multiobjetivo. Colmeias hexagonais e raios adaptativos não superaram o "
            "baseline por bairro quando faltou sinal local individualizado; esses resultados "
            "foram preservados como evidência negativa útil.",
            st["Body"],
        ),
        P(
            "Essa postura evita que uma camada visual seja promovida apenas por parecer precisa. "
            "A próxima fronteira é aprender sinais locais por célula ou rua, mantendo o bairro "
            "como camada estratégica e exigindo ganho empírico antes de qualquer adoção.",
            st["Body"],
        ),
        P("Síntese", st["H2"]),
        P(
            "A plataforma reúne processamento de dados, modelos regionais, grafos territoriais, "
            "explicabilidade, simulação, monitoramento e pesquisa reprodutível. O acesso a dados "
            "oficiais da SUPESP não inicia o produto; ele permite consolidar, validar e ampliar "
            "uma capacidade já construída, com governança e valor público mensurável.",
            st["Body"],
        ),
        Spacer(1, 0.12 * cm),
        P(
            "Base do resumo: orquestrador, processamento de dados, arquiteturas, modelos ativos "
            "e experimentais, artefatos operacionais e registros de pesquisa verificados em julho de 2026.",
            st["Small"],
        ),
    ]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()

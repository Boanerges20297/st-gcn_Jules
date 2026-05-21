import argparse
import json
import os
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


def read_optional_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def normalize_text(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).upper().strip()


def get_scope_config(scope: str) -> dict[str, str]:
    if scope == "fortaleza":
        return {"csv": "risk_fortaleza_latest.csv", "label": "Fortaleza"}
    if scope == "rmf":
        return {"csv": "risk_rmf_latest.csv", "label": "RMF"}
    if scope == "interior":
        return {"csv": "risk_interior_latest.csv", "label": "Interior"}
    return {"csv": "risk_snapshot_latest.csv", "label": "Geral"}


def get_query_terms(value: str) -> list[str]:
    stop_words = {
        "A", "O", "AS", "OS", "DE", "DA", "DO", "DAS", "DOS", "E", "EM", "NO", "NA", "NOS", "NAS",
        "UM", "UMA", "UNS", "UMAS", "POR", "PARA", "COM", "SEM", "SOBRE", "ULTIMOS", "ULTIMAS",
        "DIAS", "DIA", "MES", "MESES", "ANO", "ANOS", "ANALISE", "TATICA", "OPERACIONAL", "EVENTOS",
        "OBSERVADOS", "BASE", "DADOS", "DEPENDENDO", "SNAPSHOT", "HERMES", "MEMPALACE", "FACA", "DIGA",
        "PRINCIPAIS", "FOCOS", "PADROES", "SE", "ESTIVER", "USANDO", "APENAS", "ESSE", "CSV", "DEIXE",
        "ISSO", "EXPLICITO", "ME", "QUAL", "QUAIS", "COMO", "ONDE", "PORQUE", "QUE", "BAIRRO", "BAIRROS",
        "CIDADE", "CIDADES",
    }
    normalized = normalize_text(value)
    tokens = [token for token in __import__("re").split(r"[^A-Z0-9]+", normalized) if len(token) >= 3 and token not in stop_words]
    return list(dict.fromkeys(tokens))


def get_csv_excerpt(path: Path) -> str:
    if not path.exists():
        return ""
    columns = [
        "rank",
        "name",
        "risk_score",
        "risk_level",
        "confidence_pct",
        "expressiveness_pct",
        "top_driver_1",
        "top_driver_2",
        "leitura_rapida_gestor",
        "por_que_importa_gestor",
        "proxima_acao_gestor",
    ]
    rows = pd.read_csv(path).head(12)
    available = [column for column in columns if column in rows.columns]
    if not available:
        return ""
    return rows[available].to_csv(index=False)


def get_query_specific_context(query: str, scope_csv_path: Path, tactical_csv_path: Path, requested_scope: str) -> str:
    query_normalized = normalize_text(query)
    query_terms = get_query_terms(query)
    lines = [f"Pergunta normalizada: {query_normalized}", f"Termos extraidos: {', '.join(query_terms)}", ""]

    if scope_csv_path.exists():
        rows = pd.read_csv(scope_csv_path)
        if "name" in rows.columns:
            names = rows["name"].fillna("").astype(str).map(normalize_text)
            mask = names.map(lambda value: (query_normalized and query_normalized in value) or any(term in value for term in query_terms))
            matches = rows.loc[mask].head(8)
            if not matches.empty:
                keep = [column for column in ["rank", "name", "risk_score", "risk_level", "confidence_pct", "top_driver_1", "leitura_rapida_gestor", "por_que_importa_gestor"] if column in matches.columns]
                lines.extend(["Correspondencias no CSV oficial do escopo:", matches[keep].to_csv(index=False), ""])
            else:
                lines.extend(["Correspondencias no CSV oficial do escopo: nenhuma correspondencia direta.", ""])

    if tactical_csv_path.exists():
        rows = pd.read_csv(tactical_csv_path)
        if requested_scope == "fortaleza" and "cidade" in rows.columns:
            rows = rows.loc[rows["cidade"].fillna("").astype(str).map(normalize_text) == "FORTALEZA"]
        searchable_cols = [column for column in ["cidade", "bairro", "name", "tipo_evento"] if column in rows.columns]
        if searchable_cols:
            haystack = rows[searchable_cols].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)
            mask = haystack.map(lambda value: (query_normalized and query_normalized in value) or any(term in value for term in query_terms))
            matches = rows.loc[mask]
            if not matches.empty:
                lines.append(f"Correspondencias no CSV tatico 14d: {len(matches)} registros")
                if "bairro" in matches.columns:
                    bairro_groups = matches["bairro"].fillna("").astype(str)
                    bairro_summary = bairro_groups[bairro_groups != ""].value_counts().head(8)
                    lines.append("Bairros/cidades relacionados encontrados:")
                    lines.extend([f"- Bairro: {idx} | registros: {count}" for idx, count in bairro_summary.items()])
                if "tipo_evento" in matches.columns:
                    event_summary = matches["tipo_evento"].fillna("").astype(str)
                    event_summary = event_summary[event_summary != ""].value_counts().head(8)
                    lines.extend(["", "Padroes operacionais nas correspondencias:"])
                    lines.extend([f"- Tipo evento: {idx} | registros: {count}" for idx, count in event_summary.items()])
                sample_cols = [column for column in ["data", "cidade", "bairro", "tipo_evento", "nature", "qtd_mortes", "clima"] if column in matches.columns]
                if sample_cols:
                    lines.extend(["", "Extrato das correspondencias especificas:", matches[sample_cols].head(30).to_csv(index=False)])
            else:
                lines.extend([
                    "Correspondencias no CSV tatico 14d: nenhuma correspondencia direta.",
                    "Mesmo sem match direto, ainda existe contexto tatico agregado do escopo e ele deve ser usado para responder de forma util, sem negativa vazia.",
                ])

    return "\n".join(lines).strip()


def get_tactical_14d_context(path: Path, requested_scope: str, scope_csv_path: Path) -> str:
    if not path.exists():
        return "CSV tatico de 14 dias indisponivel."
    rows = pd.read_csv(path)
    if rows.empty:
        return "CSV tatico de 14 dias sem registros."
    scope_rows = rows.copy()
    if requested_scope == "fortaleza" and "cidade" in rows.columns:
        scope_rows = rows.loc[rows["cidade"].fillna("").astype(str).map(normalize_text) == "FORTALEZA"]
    elif requested_scope in {"rmf", "interior"} and scope_csv_path.exists():
        scope_df = pd.read_csv(scope_csv_path)
        if "name" in scope_df.columns and "cidade" in rows.columns:
            scope_names = set(scope_df["name"].fillna("").astype(str).map(normalize_text))
            scope_rows = rows.loc[rows["cidade"].fillna("").astype(str).map(normalize_text).isin(scope_names)]
    if scope_rows.empty:
        scope_rows = rows

    top_cities = []
    if "cidade" in scope_rows.columns:
        top_cities = [f"- Cidade: {idx} | registros: {count}" for idx, count in scope_rows["cidade"].fillna("").astype(str).value_counts().head(5).items() if idx]
    top_bairros = []
    if "bairro" in scope_rows.columns:
        top_bairros = [f"- Bairro: {idx} | registros: {count}" for idx, count in scope_rows["bairro"].fillna("").astype(str).value_counts().head(8).items() if idx]
    top_eventos = []
    if "tipo_evento" in scope_rows.columns:
        top_eventos = [f"- Tipo evento: {idx} | registros: {count}" for idx, count in scope_rows["tipo_evento"].fillna("").astype(str).value_counts().head(8).items() if idx]
    excerpt_cols = [column for column in ["data", "cidade", "bairro", "tipo_evento", "tipo", "nature", "qtd_mortes", "clima"] if column in scope_rows.columns]
    excerpt = scope_rows[excerpt_cols].head(80).to_csv(index=False) if excerpt_cols else ""

    lines = [
        "Fonte complementar independente: outputs/hermes/dados_status_enriquecido_14d_latest.csv",
        f"Escopo tatico considerado: {requested_scope}",
        f"Registros considerados: {len(scope_rows)}",
        "",
        "Top cidades por registros nos ultimos 14 dias:",
        "\n".join(top_cities),
        "",
        "Top bairros por registros nos ultimos 14 dias:",
        "\n".join(top_bairros),
        "",
        "Top tipos de evento nos ultimos 14 dias:",
        "\n".join(top_eventos),
        "",
        "Extrato operacional do CSV tatico 14d:",
        excerpt,
    ]
    return "\n".join(lines).strip()


def invoke_gemini_text(prompt_text: str, working_directory: Path, model_name: str) -> dict[str, str]:
    instruction = "Responda usando apenas o contexto recebido via stdin. Seja objetivo, analitico e fiel aos artefatos recebidos."
    command = ["gemini", "--skip-trust", "--output-format", "text"]
    if model_name:
        command.extend(["-m", model_name])
    command.extend(["-p", instruction])

    completed = subprocess.run(
        command,
        cwd=working_directory,
        input=prompt_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )

    if completed.returncode != 0 and model_name:
        fallback = subprocess.run(
            ["gemini", "--skip-trust", "--output-format", "text", "-p", instruction],
            cwd=working_directory,
            input=prompt_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        if fallback.returncode == 0 and fallback.stdout.strip():
            return {"text": fallback.stdout.strip(), "model_used": "default-cli", "stderr": fallback.stderr.strip()}
        raise RuntimeError(f"Gemini CLI falhou com '{model_name}' e no fallback. STDERR inicial: {completed.stderr}\nSTDERR fallback: {fallback.stderr}")

    if completed.returncode != 0:
        raise RuntimeError(f"Gemini CLI falhou: {completed.stderr}")
    if not completed.stdout.strip():
        raise RuntimeError(f"Gemini CLI nao retornou conteudo util. STDERR: {completed.stderr}")
    return {"text": completed.stdout.strip(), "model_used": model_name or "default-cli", "stderr": completed.stderr.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta Gemini CLI com memoria operacional MemPalace")
    parser.add_argument("--query", required=True)
    parser.add_argument("--scope", default="fortaleza", choices=["fortaleza", "rmf", "interior", "geral"])
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--context-root", default="")
    parser.add_argument("--chat-dir", default="")
    parser.add_argument("--chat-id", default="", help="Optional chat identifier to isolate latest answer files")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    context_root = Path(args.context_root).resolve() if args.context_root else project_root
    scope_config = get_scope_config(args.scope)
    outputs_dir = project_root / "outputs" / "hermes"
    chat_dir = Path(args.chat_dir).resolve() if args.chat_dir else project_root / "outputs" / "mempalace" / "chat"
    history_dir = chat_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    brief_path = outputs_dir / "risk_brief_latest.md"
    csv_path = outputs_dir / scope_config["csv"]
    tactical_14d_path = outputs_dir / "dados_status_enriquecido_14d_latest.csv"
    tactical_14d_summary_md_path = outputs_dir / "dados_status_enriquecido_14d_summary_latest.md"
    tactical_14d_summary_json_path = outputs_dir / "dados_status_enriquecido_14d_summary_latest.json"
    mempalace_path = project_root / ".mempalace.md"
    legacy_directives_path = project_root / ".hermes.md"
    soul_path = context_root / ".mempalace" / "SOUL.md"

    soul_text = read_optional_file(soul_path)
    mempalace_text = read_optional_file(mempalace_path)
    legacy_directives = read_optional_file(legacy_directives_path)
    brief_text = read_optional_file(brief_path)
    csv_excerpt = get_csv_excerpt(csv_path)
    tactical_14d_summary_md = read_optional_file(tactical_14d_summary_md_path)
    tactical_14d_summary_json = read_optional_file(tactical_14d_summary_json_path)
    query_specific_context = get_query_specific_context(args.query, csv_path, tactical_14d_path, args.scope)
    tactical_14d_context = tactical_14d_summary_md or get_tactical_14d_context(tactical_14d_path, args.scope, csv_path)

    prompt = f"""
Voce esta respondendo como um assistente operacional do projeto Report Preview com memoria MemPalace.

Objetivo:
- responder a pergunta do usuario com base nos artefatos do projeto;
- manter estilo pt-BR, objetivo e analitico;
- usar leitura gerencial e identificar padroes quando existirem;
- entregar sempre uma previsao operacional para os proximos 7 dias, nunca uma resposta apenas retrospectiva;
- nao inventar rankings, nomes ou causalidade fora do contexto recebido;
- se os artefatos principais nao trouxerem referencia suficiente para responder diretamente, usar obrigatoriamente o CSV tatico dos ultimos 14 dias para construir uma analise independente;
- priorizar o resumo tatico 14d pre-calculado quando ele existir, porque ele ja agrega focos territoriais e padroes operacionais do CSV bruto;
- se houver dados uteis no CSV tatico 14d, nunca responder com ausencia total de base;
- quando a resposta vier principalmente do CSV tatico 14d, explicitar que se trata de projecao tatico-operacional para os proximos 7 dias sustentada pelos ultimos 14 dias e que ela nao incorpora necessariamente os artefatos preditivos mais recentes;
- se a pergunta mencionar um bairro, cidade ou localidade especifica, usar obrigatoriamente a secao ALVO ESPECIFICO DA PERGUNTA antes de concluir que nao ha referencia;
- se nao houver match direto da localidade no CSV tatico 14d, ainda assim responder com a melhor leitura tatica do escopo e dizer explicitamente apenas que nao houve correspondencia direta da localidade na janela, sem encerrar a resposta nisso.

Regras obrigatorias de previsao:
- toda resposta deve orientar a decisao para os proximos 7 dias;
- usar primeiro os snapshots e artefatos oficiais do projeto quando houver ranking, score, driver ou localidade no snapshot atual;
- usar o tatico 14d como sustentacao, aceleracao, concentracao territorial e padrao operacional para a projecao dos proximos 7 dias;
- quando houver localidade especifica no snapshot, transformar score, driver e contexto tatico em previsao objetiva para os proximos 7 dias;
- quando nao houver localidade especifica no snapshot ou no CSV tatico, ainda assim produzir previsao para os proximos 7 dias com base no melhor contexto territorial do escopo, deixando explicita a limitacao da localidade;
- nunca responder apenas que nao encontrou dados; sempre devolver uma previsao operacional util para os proximos 7 dias.

Formato preferido:
1. Dados ate
2. Fonte
3. Leitura rapida
4. Previsao para os proximos 7 dias
5. Por que importa
6. Proxima acao

MEMORIA GLOBAL MEMPALACE (SOUL):
{soul_text}

MEMORIA LOCAL DO PROJETO (.mempalace.md):
{mempalace_text}

DIRETIVAS LEGADAS DO PROJETO (compatibilidade):
{legacy_directives}

BRIEF EXECUTIVO DO PROJETO:
{brief_text}

EXTRATO CSV DO ESCOPO {scope_config['label']}:
{csv_excerpt}

ALVO ESPECIFICO DA PERGUNTA:
{query_specific_context}

CONTEXTO TATICO INDEPENDENTE DOS ULTIMOS 14 DIAS:
{tactical_14d_context}

RESUMO TATICO 14D EM JSON (quando precisar de estrutura):
{tactical_14d_summary_json}

PERGUNTA DO USUARIO:
{args.query}
""".strip()

    result = invoke_gemini_text(prompt, project_root, args.gemini_model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_answer_path = chat_dir / f"gemini_chat_{args.scope}_latest.md"
    # If chat-id provided, create per-chat latest file to avoid cross-chat leakage
    if args.chat_id:
        latest_answer_path = chat_dir / f"gemini_chat_{args.scope}_{args.chat_id}_latest.md"
    history_answer_path = history_dir / f"gemini_chat_{args.scope}_{timestamp}.md"
    history_prompt_path = history_dir / f"gemini_chat_prompt_{args.scope}_{timestamp}.txt"

    content = "\n".join(
        [
            "# Resposta Gemini com Memoria MemPalace",
            "",
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Escopo: {scope_config['label']}",
            f"Modelo Gemini CLI: {result['model_used']}",
            f"Pergunta: {args.query}",
            f"CSV base: outputs/hermes/{scope_config['csv']}",
            "CSV tatico 14d: outputs/hermes/dados_status_enriquecido_14d_latest.csv",
            "",
            result["text"],
            "",
        ]
    )

    latest_answer_path.write_text(content, encoding="utf-8")
    history_answer_path.write_text(content, encoding="utf-8")
    history_prompt_path.write_text(prompt, encoding="utf-8")

    print(f"Resposta salva em: {latest_answer_path}")
    print(f"Historico da resposta: {history_answer_path}")
    print(f"Prompt salvo em: {history_prompt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
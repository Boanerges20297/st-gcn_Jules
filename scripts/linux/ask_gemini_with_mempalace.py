import argparse
import json
import os
import subprocess
import sys
import time
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


def is_polluted_or_trivial(insight: str) -> bool:
    if not insight:
        return True
    insight_lower = insight.lower().strip()
    normalized = "".join(ch for ch in unicodedata.normalize("NFKD", insight_lower) if not unicodedata.combining(ch))
    
    # Reject specific test/polluted prompts
    if "capital e fortaleza" in normalized or "capital do ceara" in normalized:
        return True
    if normalized.startswith("a capital") or normalized.startswith("capital "):
        return True
        
    # Reject extremely short entries
    if len(insight.strip()) < 10:
        return True
        
    # Reject known non-operational test phrases
    non_operational_patterns = ["fortaleza e a capital", "fortaleza e capital", "capital e fortaleza"]
    if any(pat in normalized for pat in non_operational_patterns):
        return True
        
    return False


def get_global_learnings(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            return ""
        valid_entries = []
        for entry in data:
            if isinstance(entry, dict) and entry.get("tactical_insight"):
                insight = entry["tactical_insight"].strip()
                if is_polluted_or_trivial(insight):
                    continue
                valid_entries.append(entry)
        if not valid_entries:
            return ""
        lines = [
            "## CONHECIMENTO COLETIVO EM EVOLUÇÃO (APRENDIZADOS DO TIME)",
            "Estes são aprendizados operacionais reais, anomalias empíricas de dados e críticas táticas ao modelo preditivo identificadas recentemente por outros analistas da equipe em campo. Considere-as ativamente ao formular e calibrar suas análises/projeções para os próximos 7 a 14 dias:",
        ]
        seen = set()
        for entry in reversed(valid_entries):
            scope = entry.get("scope", "geral").upper()
            topic = entry.get("topic", "N/A")
            insight = entry.get("tactical_insight", "").strip()
            key = (scope, topic, insight)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- [Escopo: {scope} | Alvo: {topic}]: {insight}")
            if len(seen) >= 15:
                break
        return "\n".join(lines).strip()
    except Exception as e:
        print(f"Erro ao ler aprendizados coletivos: {e}", file=sys.stderr)
        return ""


def save_learning(path: Path, entry: dict):
    if not entry or not entry.get("tactical_insight"):
        return
    if is_polluted_or_trivial(entry["tactical_insight"]):
        print(f"Ignorando aprendizado poluído/trivial: {entry['tactical_insight']}")
        return

    for attempt in range(5):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing_data = []
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = []
                except Exception:
                    existing_data = []
            
            duplicate = False
            for old_entry in existing_data[-5:]:
                if old_entry.get("tactical_insight") == entry["tactical_insight"]:
                    duplicate = True
                    break
            if duplicate:
                return
                
            existing_data.append(entry)
            if len(existing_data) > 100:
                existing_data = existing_data[-100:]
                
            import tempfile
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=path.parent) as temp:
                json.dump(existing_data, temp, ensure_ascii=False, indent=2)
                temp_name = temp.name
            
            os.replace(temp_name, path)
            print(f"Aprendizado coletivo salvo: {entry['tactical_insight']}")
            break
        except Exception as e:
            print(f"Tentativa {attempt+1} falhou ao salvar aprendizado: {e}", file=sys.stderr)
            time.sleep(0.1 * (attempt + 1))


def run_learning_extractor(query: str, answer: str, project_root: Path, model_name: str, global_learnings_path: Path):
    instruction = (
        "Você é o analista sênior do Report Preview. Com base na pergunta do usuário e na resposta tática/crítica ao modelo gerada, extraia um único aprendizado ou anomalia operacional relevante para o time.\n"
        "ATENÇÃO: NUNCA inclua dados pessoais, identificadores de chat, ids ou nomes de usuários. O aprendizado deve ser 100% anônimo e puramente tático (ex: 'Bairro X apresentou trégua nos homicídios contrariando score de risco alto').\n"
        "Responda EXCLUSIVAMENTE em formato JSON puro, sem markdown, contendo:\n"
        "{\n"
        "  \"scope\": \"fortaleza/rmf/interior/geral\",\n"
        "  \"topic\": \"Nome do Bairro, Cidade ou Facção analisada\",\n"
        "  \"tactical_insight\": \"A síntese da crítica ou fato empírico extraído\"\n"
        "}\n"
        "Se a interação foi genérica ou não gerou nenhuma nova lição de dados ou crítica relevante para o time, retorne:\n"
        "{\n"
        "  \"scope\": \"\",\n"
        "  \"topic\": \"\",\n"
        "  \"tactical_insight\": \"\"\n"
        "}"
    )
    
    prompt_input = f"PERGUNTA DO USUÁRIO:\n{query}\n\nRESPOSTA OPERACIONAL DO REPORT PREVIEW:\n{answer}"
    
    command = ["gemini", "--skip-trust", "--output-format", "text"]
    if model_name:
        command.extend(["-m", model_name])
    command.extend(["-p", instruction])
    
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            input=prompt_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        if completed.returncode != 0:
            print(f"Learning extractor falhou: {completed.stderr}", file=sys.stderr)
            return
        
        raw_output = completed.stdout.strip()
        if raw_output.startswith("```"):
            lines = raw_output.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                raw_output = "\n".join(lines[1:-1]).strip()
        
        data = json.loads(raw_output)
        insight = data.get("tactical_insight", "").strip()
        scope = data.get("scope", "").strip()
        topic = data.get("topic", "").strip()
        
        if not insight:
            return
            
        save_learning(global_learnings_path, {
            "timestamp": datetime.now().isoformat(),
            "scope": scope,
            "topic": topic,
            "tactical_insight": insight
        })
    except Exception as e:
        print(f"Erro ao extrair aprendizado: {e}", file=sys.stderr)


def get_query_terms(value: str) -> list[str]:
    stop_words = {
        "A", "O", "AS", "OS", "DE", "DA", "DO", "DAS", "DOS", "E", "EM", "NO", "NA", "NOS", "NAS",
        "UM", "UMA", "UNS", "UMAS", "POR", "PARA", "COM", "SEM", "SOBRE", "ULTIMOS", "ULTIMAS",
        "DIAS", "DIA", "MES", "MESES", "ANO", "ANOS", "ANALISE", "TATICA", "OPERACIONAL", "EVENTOS",
        "OBSERVADOS", "BASE", "DADOS", "DEPENDENDO", "SNAPSHOT", "HERMES", "MEMPALACE", "FACA", "DIGA",
        "PRINCIPAIS", "FOCOS", "PADROES", "SE", "ESTIVER", "USANDO", "APENAS", "ESSE", "CSV", "DEIXE",
        "ISSO", "EXPLICITO", "ME", "QUAL", "QUAIS", "COMO", "ONDE", "PORQUE", "QUE", "BAIRRO", "BAIRROS",
        "CIDADE", "CIDADES", "SAO",
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


def get_query_specific_context(query: str, scope_csv_path: Path, tactical_csv_path: Path, micronodes_csv_path: Path, requested_scope: str) -> str:
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

    if micronodes_csv_path.exists():
        m_rows = pd.read_csv(micronodes_csv_path)
        if requested_scope == "fortaleza":
            m_rows = m_rows.loc[m_rows["regional"].fillna("").astype(str).map(normalize_text) == "CAPITAL"]
        elif requested_scope == "rmf":
            m_rows = m_rows.loc[m_rows["regional"].fillna("").astype(str).map(normalize_text) == "RMF"]
        elif requested_scope == "interior":
            m_rows = m_rows.loc[m_rows["regional"].fillna("").astype(str).map(normalize_text) == "INTERIOR"]
            
        searchable_cols = [col for col in ["micronode_id", "bairro", "faction", "nearby_streets"] if col in m_rows.columns]
        if searchable_cols:
            haystack = m_rows[searchable_cols].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)
            mask = haystack.map(lambda value: (query_normalized and query_normalized in value) or any(term in value for term in query_terms))
            matches = m_rows.loc[mask]
            if not matches.empty:
                keep = [col for col in ["global_rank", "micronode_id", "score", "bairro", "regional", "faction", "local_street_pressure", "nearby_streets"] if col in matches.columns]
                lines.extend([
                    "MICRONODOS CRITICOS DA LOCALIDADE (SENTINELA):",
                    matches[keep].head(15).to_csv(index=False),
                    ""
                ])
            else:
                lines.extend(["MICRONODOS CRITICOS DA LOCALIDADE (SENTINELA): nenhuma correspondencia direta.", ""])

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
    instruction = (
        "Você é o analista de inteligência tático-operacional do projeto Report Preview. "
        "NUNCA cite os nomes de arquivos internos, tabelas CSV, diretórios como 'outputs/hermes', 'caminho_crime.csv', ou nomenclaturas como 'Hermes', 'CPRAIO' em suas respostas. "
        "Refira-se à fonte de dados e inteligência do sistema sempre e apenas como 'Report Preview'. "
        "Responda em PT-BR com um tom tático mas natural, agindo como um analista de dados explicando de forma direta ao gestor o que faz mais sentido. "
        "Seu limite de resposta padrão é de até **10 linhas** de leitura acionável. "
        "EXCEÇÃO: Se for necessário expor um ranking organizado, você tem um limite estendido de no máximo **10 linhas para o ranking estruturado** mais **5 linhas expositivas ou analíticas** (total máximo absoluto de 15 linhas). "
        "Como você tem acesso aos dados operacionais brutos, VOCÊ PODE E DEVE CRITICAR AS DECISÕES/PROJEÇÕES DO MODELO SE ACHAR NECESSÁRIO, gerando sua própria análise tática alternativa para os próximos 14 dias se detectar incoerências. "
        "Ao receber perguntas mais superficiais, use de discussão saudável, técnica e analítica com o gestor para instigar um debate estratégico produtivo. "
        "Para rankings explícitos: liste até 5 focos + por que importa + próxima ação. "
        "Para perguntas táticas: traga dados diretos + drivers territoriais/operacionais + crítica do modelo (se aplicável) + próxima ação acionável."
    )
    
    # Load all available API keys for rotation
    api_keys = []
    
    # 1. From environment variables
    for env_key in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
        val = os.environ.get(env_key)
        if val and val not in api_keys:
            api_keys.append(val)
    # 2. Parse from .env in working directory to ensure all are loaded
    dotenv_path = working_directory / ".env"
    if dotenv_path.exists():
        try:
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and line.startswith("GEMINI_API_KEY"):
                    k, v = line.split("=", 1)
                    v_clean = v.strip().strip('"').strip("'")
                    if v_clean and v_clean not in api_keys:
                        api_keys.append(v_clean)
        except Exception as e:
            print(f"Erro ao ler .env para obter chaves: {e}", file=sys.stderr)

    # 3. Add standard system-wide Google Auth credentials (None) at the very end as absolute fallback
    if None not in api_keys:
        api_keys.append(None)

    last_error = ""
    # Try invoking Gemini with key rotation
    for idx, api_key in enumerate(api_keys):
        env = os.environ.copy()
        if api_key:
            env["GEMINI_API_KEY"] = api_key
            print(f"Tentando Gemini CLI com chave de API {idx + 1}/{len(api_keys)-1}...", file=sys.stderr)
        else:
            env.pop("GEMINI_API_KEY", None)
            print("Tentando Gemini CLI com credenciais globais do sistema (Google Auth)...", file=sys.stderr)
            
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
            env=env
        )

        if completed.returncode == 0 and completed.stdout.strip():
            return {
                "text": completed.stdout.strip(),
                "model_used": model_name or "default-cli",
                "stderr": completed.stderr.strip()
            }
            
        # If it failed, record stderr and try with the next key if any
        err_msg = completed.stderr.strip() or completed.stdout.strip() or "Erro desconhecido"
        label = "Credenciais do Sistema" if api_key is None else f"Chave {idx + 1}"
        print(f"{label} falhou. Erro: {err_msg[:200]}...", file=sys.stderr)
        last_error += f"[{label}]: {err_msg}\n"
        
    # If we had fallback or model name was specified, try without model name and rotate again!
    if model_name:
        print("Todas as chaves falharam com o modelo especificado. Tentando sem modelo (fallback)...", file=sys.stderr)
        for idx, api_key in enumerate(api_keys):
            env = os.environ.copy()
            if api_key:
                env["GEMINI_API_KEY"] = api_key
            else:
                env.pop("GEMINI_API_KEY", None)
                
            command = ["gemini", "--skip-trust", "--output-format", "text", "-p", instruction]
            completed = subprocess.run(
                command,
                cwd=working_directory,
                input=prompt_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
                env=env
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return {
                    "text": completed.stdout.strip(),
                    "model_used": "default-cli",
                    "stderr": completed.stderr.strip()
                }
            err_msg = completed.stderr.strip() or completed.stdout.strip() or "Erro desconhecido"
            label = "Credenciais do Sistema" if api_key is None else f"Chave {idx + 1}"
            last_error += f"[Fallback {label}]: {err_msg}\n"

    raise RuntimeError(f"Gemini CLI falhou com todas as chaves carregadas.\nDetalhes dos erros:\n{last_error}")


def clean_response_pollution(text: str, query: str) -> str:
    if not text:
        return text
        
    query_lower = query.lower()
    if "capital" in query_lower:
        return text
        
    import unicodedata
    import re
    
    norm_chars = []
    orig_indices = []
    
    for orig_idx, char in enumerate(text):
        decomp = unicodedata.normalize('NFKD', char)
        stripped_decomp = "".join(c for c in decomp if not unicodedata.combining(c))
        for c in stripped_decomp:
            norm_chars.append(c)
            orig_indices.append(orig_idx)
            
    norm_text = "".join(norm_chars).lower()
    
    suffix_pattern = r"([,;.]?\s*(onde\b|que\b|e\s+o\s+nucleo\b|de\s+modo\s+que\b)|[,;.:]?\s*)\s*"
    pattern_std = r"(como\s+)?a\s+capital\s+(do\s+ceara\s+)?(e\s+|eh\s+)?fortaleza\b" + suffix_pattern
    pattern_inv = r"fortaleza\s+(e\s+|eh\s+)?(a\s+)?capital\s*(do\s+ceara\s*)?\b" + suffix_pattern
    
    match = re.search(pattern_std, norm_text)
    if not match:
        match = re.search(pattern_inv, norm_text)
        
    if match:
        norm_start_idx = match.start()
        norm_end_idx = match.end()
        
        orig_start_idx = orig_indices[norm_start_idx]
        
        if norm_end_idx < len(orig_indices):
            orig_end_idx = orig_indices[norm_end_idx]
        else:
            orig_end_idx = len(text)
            
        left_part = text[:orig_start_idx].rstrip()
        right_part = text[orig_end_idx:].lstrip()
        
        if left_part and right_part:
            if left_part.endswith("|") or left_part.endswith(":") or left_part.endswith("—") or left_part.endswith("-"):
                cleaned_right = right_part[0].upper() + right_part[1:] if right_part else ""
                return f"{left_part} {cleaned_right}".strip()
            else:
                cleaned_right = right_part[0].upper() + right_part[1:] if right_part else ""
                return left_part + " " + cleaned_right
        elif right_part:
            return right_part[0].upper() + right_part[1:]
        elif left_part:
            return left_part
            
    return text


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
    micronodes_csv_path = outputs_dir / "visible_micronodes.csv"
    tactical_14d_summary_md_path = outputs_dir / "dados_status_enriquecido_14d_summary_latest.md"
    tactical_14d_summary_json_path = outputs_dir / "dados_status_enriquecido_14d_summary_latest.json"
    mempalace_path = project_root / ".mempalace.md"
    legacy_directives_path = project_root / ".hermes.md"
    soul_path = context_root / ".mempalace" / "SOUL.md"
    global_learnings_path = project_root / "outputs" / "mempalace" / "global_learnings.json"

    soul_text = read_optional_file(soul_path)
    mempalace_text = read_optional_file(mempalace_path)
    legacy_directives = read_optional_file(legacy_directives_path)
    brief_text = read_optional_file(brief_path)
    csv_excerpt = get_csv_excerpt(csv_path)
    tactical_14d_summary_md = read_optional_file(tactical_14d_summary_md_path)
    tactical_14d_summary_json = read_optional_file(tactical_14d_summary_json_path)
    query_specific_context = get_query_specific_context(args.query, csv_path, tactical_14d_path, micronodes_csv_path, args.scope)
    tactical_14d_context = tactical_14d_summary_md or get_tactical_14d_context(tactical_14d_path, args.scope, csv_path)
    global_learnings_text = get_global_learnings(global_learnings_path)

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

Formato de resposta — use o correto para o tipo de pergunta:

Se a pergunta pedir RANKING explicito (top/lista/posicao):
  Dados ate DD/MM/AAAA | Fonte: Report Preview
  [ate 5 bullets: local — score — nivel — driver]
  Por que importa: [1 frase]
  Proxima acao: [1 frase]

Se a pergunta for TATICA (localidade, faccao, evento, driver, risco pontual):
  [Localidade/assunto]: score X | nivel Y
  Drivers: A, B
  Ultimos 14d: [fato direto]
  Proxima acao: [1 frase]

Nao use o formato de ranking para perguntas taticas.

MEMORIA GLOBAL MEMPALACE (SOUL):
{soul_text}

MEMORIA LOCAL DO PROJETO (.mempalace.md):
{mempalace_text}

DIRETIVAS LEGADAS DO PROJETO (compatibilidade):
{legacy_directives}

BRIEF EXECUTIVO DO PROJETO:
{brief_text}

{global_learnings_text}

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
    result["text"] = clean_response_pollution(result["text"], args.query)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_answer_path = chat_dir / f"gemini_chat_{args.scope}_latest.md"
    if args.chat_id:
        latest_answer_path = chat_dir / f"gemini_chat_{args.scope}_{args.chat_id}_latest.md"
    history_answer_path = history_dir / f"gemini_chat_{args.scope}_{timestamp}.md"
    history_prompt_path = history_dir / f"gemini_chat_prompt_{args.scope}_{timestamp}.txt"
    query_line = args.query.replace("\r\n", " ").replace("\n", " ")

    content = "\n".join(
        [
            "# Resposta - Report Preview",
            "",
            f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Escopo: {scope_config['label']}",
            f"Modelo: {result['model_used']}",
            f"Pergunta: {query_line}",
            f"Fonte: Report Preview",
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

    run_learning_extractor(args.query, result["text"], project_root, args.gemini_model, global_learnings_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

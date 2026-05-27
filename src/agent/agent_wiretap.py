import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentWiretap:
    """
    Audit Log / Interceptador de Diálogos para o Sistema Multi-Agente.
    Permite ouvir as interações confidenciais entre o Gerente Geral e os especialistas blindados.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "agents_dialogue.log")
        self.json_file = os.path.join(log_dir, "agents_dialogue_history.json")
        os.makedirs(self.log_dir, exist_ok=True)

    def check_and_cleanup_logs(self):
        """
        Verifica a data das últimas mensagens registradas no JSON.
        Se houver logs mais antigos que 7 dias, faz o backup/arquivamento deles
        em logs/archives/ e limpa o arquivo ativo para que ele permaneça enxuto e não infle.
        """
        if not os.path.exists(self.json_file):
            return

        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=7)
        history_data = []

        try:
            with open(self.json_file, "r", encoding="utf-8") as jf:
                history_data = json.load(jf)
        except Exception:
            return

        if not history_data:
            return

        active_entries = []
        archived_entries = []
        has_changes = False

        for entry in history_data:
            try:
                entry_dt = datetime.strptime(entry.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
                if entry_dt >= cutoff_date:
                    active_entries.append(entry)
                else:
                    archived_entries.append(entry)
                    has_changes = True
            except Exception:
                active_entries.append(entry)

        if has_changes:
            # 1. Se houver itens antigos, salva em um arquivo de backup histórico datado
            if archived_entries:
                archive_dir = os.path.join(self.log_dir, "archives")
                os.makedirs(archive_dir, exist_ok=True)
                archive_file = os.path.join(archive_dir, f"dialogue_archive_{cutoff_date.strftime('%Y-%m-%d')}.json")
                try:
                    # Carrega se já existir arquivo desse dia para anexar
                    existing_archive = []
                    if os.path.exists(archive_file):
                        with open(archive_file, "r", encoding="utf-8") as arf:
                            existing_archive = json.load(arf)
                    
                    existing_archive.extend(archived_entries)
                    with open(archive_file, "w", encoding="utf-8") as arf:
                        json.dump(existing_archive, arf, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Erro ao salvar arquivo histórico de diálogos: {e}")

            # 2. Atualizar o JSON ativo principal (que fica sempre enxuto)
            try:
                with open(self.json_file, "w", encoding="utf-8") as jf:
                    json.dump(active_entries, jf, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Erro ao salvar JSON ativo de diálogos: {e}")

            # 3. Limpar e re-gerar o log visual (.log) para conter apenas a janela móvel de 7 dias
            try:
                divider = "=" * 80
                with open(self.log_file, "w", encoding="utf-8") as lf:
                    lf.write("=== Janela Móvel de Log: Últimos 7 Dias (Mais antigos arquivados) ===\n")
                    for entry in active_entries:
                        log_entry = (
                            f"\n{divider}\n"
                            f"🕒 [{entry.get('timestamp')}] INTERCEPTAÇÃO DE DIÁLOGO\n"
                            f"📢 REMETENTE:   {entry.get('sender').upper()}\n"
                            f"👤 DESTINATÁRIO: {entry.get('receiver').upper()}\n"
                            f"{'-' * 40}\n"
                            f"💬 PROMPT ENVIADO:\n{entry.get('prompt').strip()}\n"
                            f"{'-' * 40}\n"
                            f"📥 RESPOSTA DO ESPECIALISTA:\n{entry.get('response').strip()}\n"
                            f"{divider}\n"
                        )
                        lf.write(log_entry)
                print("🧹 [AgentWiretap] Rotação completa! Logs antigos movidos para pasta 'archives'.")
            except Exception as e:
                logger.error(f"Erro ao re-gerar .log após arquivamento: {e}")

    def record_interaction(self, sender: str, receiver: str, prompt: str, response: str):
        """
        Registra uma conversa individual entre dois agentes tanto em formato visualmente
        legível (.log) quanto estruturado (.json).
        """
        # Executa limpeza preventiva antes de gravar a nova interação
        self.check_and_cleanup_logs()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Gravar no log visualmente polido (estilo chat)
        divider = "=" * 80
        log_entry = (
            f"\n{divider}\n"
            f"🕒 [{timestamp}] INTERCEPTAÇÃO DE DIÁLOGO\n"
            f"📢 REMETENTE:   {sender.upper()}\n"
            f"👤 DESTINATÁRIO: {receiver.upper()}\n"
            f"{'-' * 40}\n"
            f"💬 PROMPT ENVIADO:\n{prompt.strip()}\n"
            f"{'-' * 40}\n"
            f"📥 RESPOSTA DO ESPECIALISTA:\n{response.strip()}\n"
            f"{divider}\n"
        )
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as lf:
                lf.write(log_entry)
        except Exception as e:
            logger.error(f"Falha ao escrever no log do wiretap: {e}")

        # 2. Gravar no histórico JSON estruturado
        history_data = []
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, "r", encoding="utf-8") as jf:
                    history_data = json.load(jf)
            except Exception:
                history_data = []

        history_entry = {
            "timestamp": timestamp,
            "sender": sender,
            "receiver": receiver,
            "prompt": prompt,
            "response": response
        }
        history_data.append(history_entry)

        try:
            with open(self.json_file, "w", encoding="utf-8") as jf:
                json.dump(history_data, jf, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Falha ao escrever no JSON do wiretap: {e}")

# Instância Singleton padrão
wiretap = AgentWiretap()

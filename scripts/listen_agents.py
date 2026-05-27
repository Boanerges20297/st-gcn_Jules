import time
import os
import sys

# Garantir acesso ao root do projeto para importação
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

LOG_FILE = "logs/agents_dialogue.log"

def main():
    print("=" * 80)
    print("🕵️‍♂️ MONITOR DE DIÁLOGO MULTI-AGENTE (WIRETAP) — TEMPO REAL")
    print("Ouvindo as comunicações de malha fechada em background...")
    print(f"Monitorando: {LOG_FILE}")
    print("Pressione Ctrl+C para encerrar o monitoramento.")
    print("=" * 80)

    # Se o arquivo não existir ainda, aguarda ou cria
    if not os.path.exists(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=== Início das Gravações do Monitor ===\n")

    # Remover sinalizador de desligamento anterior se existir
    SHUTDOWN_FILE = "logs/.app_shutdown"
    if os.path.exists(SHUTDOWN_FILE):
        try: os.remove(SHUTDOWN_FILE)
        except Exception: pass

    # tail -f em Python
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            # Exibir as últimas 40 linhas do histórico para evitar tela em branco no início
            linhas_existentes = f.readlines()
            if linhas_existentes:
                print("\n📜 [Monitor] Mostrando as últimas 40 linhas do histórico de diálogos:")
                for linha in linhas_existentes[-40:]:
                    sys.stdout.write(linha)
                sys.stdout.write("\n")
                print("📢 [Monitor] --- FIM DO HISTÓRICO ATUAL / AGUARDANDO NOVOS DIÁLOGOS EM TEMPO REAL ---\n\n")
                sys.stdout.flush()

            # Ir para o fim do arquivo para escutar novas entradas
            f.seek(0, os.SEEK_END)
            
            while True:
                # Checar se a aplicação Flask enviou o sinal de shutdown
                if os.path.exists(SHUTDOWN_FILE):
                    print("\n\n🔌 [Monitor] Conexão perdida: A aplicação Report Preview foi encerrada.")
                    print("Desligando monitor de diálogos automaticamente...")
                    try: os.remove(SHUTDOWN_FILE)
                    except Exception: pass
                    break

                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n🕵️‍♂️ Monitoramento encerrado pelo usuário.")

if __name__ == "__main__":
    main()

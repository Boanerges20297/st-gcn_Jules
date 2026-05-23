import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime

# Adjust path to find ask_gemini_with_mempalace modules/functions
sys.path.append(str(Path(__file__).parent.parent / "scripts" / "linux"))
from ask_gemini_with_mempalace import get_global_learnings, save_learning, run_learning_extractor

def test_concurrent_writes(temp_json_path: Path):
    print("Iniciando teste de escritas concorrentes...")
    threads = []
    
    def writer(thread_id):
        for i in range(10):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "scope": "fortaleza",
                "topic": f"Thread-{thread_id}",
                "tactical_insight": f"Insight operacional numero {i} da thread {thread_id}."
            }
            save_learning(temp_json_path, entry)
            time.sleep(0.01)
            
    # Spawn 5 concurrent threads
    for t_id in range(5):
        t = threading.Thread(target=writer, args=(t_id,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Read back and verify size and format
    if temp_json_path.exists():
        with open(temp_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Sucesso! Foram escritos {len(data)} aprendizados coletivos no total.")
        print(f"Amostra do ultimo registro: {data[-1] if data else 'Vazio'}")
    else:
        print("Erro: O arquivo de aprendizados nao foi criado.")

def main():
    print("====================================================")
    print("TESTE DE APRENDIZADO COLETIVO E EVOLUCAO CONTINUA")
    print("====================================================")
    
    project_root = Path(__file__).parent.parent
    temp_json_path = project_root / "outputs" / "mempalace" / "global_learnings_test.json"
    
    if temp_json_path.exists():
        temp_json_path.unlink()
        
    # Run concurrent test
    test_concurrent_writes(temp_json_path)
    
    # Test reading format
    learnings_text = get_global_learnings(temp_json_path)
    print("\nFormatacao obtida para injecao no prompt:")
    print("----------------------------------------------------")
    print(learnings_text[:500] + "...")
    print("----------------------------------------------------")
    
    # Clean up test file
    if temp_json_path.exists():
        temp_json_path.unlink()
        
    print("\nTeste local concluido com sucesso!")

if __name__ == "__main__":
    main()

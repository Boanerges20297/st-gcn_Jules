import numpy as np
import os
import json
from datetime import datetime

class TrainingVault:
    """
    Cofre de Memória Dinâmica para Treinamento.
    Armazena o 'conhecimento' acumulado entre épocas para injetar no Canal 38.
    """
    def __init__(self, num_nodes, base_dir):
        self.num_nodes = num_nodes
        self.vault_path = os.path.join(base_dir, "data", "training_vault")
        os.makedirs(self.vault_path, exist_ok=True)
        
        # Memória de Longo Prazo (Persistência desde 2022)
        self.long_term_memory = np.zeros(num_nodes, dtype=np.float32)
        
        # Memória de Curto Prazo (Surpresas da época anterior)
        self.epoch_surprises = np.zeros(num_nodes, dtype=np.float32)
        
        # Contador de Hits
        self.hits = np.zeros(num_nodes, dtype=np.int32)

    def record_surprise(self, node_idx, intensity):
        """Registra uma falha do modelo em prever um crime real."""
        self.epoch_surprises[node_idx] += intensity
        self.long_term_memory[node_idx] += intensity * 0.1 # Acúmulo lento
        self.hits[node_idx] += 1

    def get_memory_vector(self):
        """Retorna o vetor de memória normalizado para o Canal 38."""
        # Combina Memória de Longo Prazo com as Surpresas Recentes
        combined = self.long_term_memory + self.epoch_surprises
        if combined.max() > 0:
            return combined / combined.max()
        return combined

    def clear_epoch(self):
        """Limpa as surpresas da época para iniciar um novo ciclo de observação."""
        self.epoch_surprises.fill(0)

    def save(self, epoch):
        """Salva o estado da memória para persistência."""
        state = {
            "epoch": epoch,
            "long_term": self.long_term_memory.tolist(),
            "hits": self.hits.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        path = os.path.join(self.vault_path, f"vault_state_e{epoch}.json")
        with open(path, 'w') as f:
            json.dump(state, f)

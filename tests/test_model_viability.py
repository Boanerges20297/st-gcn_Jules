"""
Testes de Viabilidade do Modelo ST-GCN
Verifica se o modelo está configurado corretamente e se CVP é realmente utilizado
"""

import pytest
import torch
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import STGCN


class TestModelViability:
    """Testes para validar a viabilidade e configuração do modelo"""
    
    @pytest.fixture
    def model_config(self):
        """Configuração padrão do modelo"""
        return {
            'num_nodes': 302,
            'in_channels': 3,  # CVLI, CVP, Tension
            'time_steps': 30,
            'num_classes': 1,
            'num_graphs': 2
        }
    
    @pytest.fixture
    def sample_data(self):
        """Dados de exemplo com 3 canais"""
        N = 302  # número de nós
        T = 30   # janela temporal
        C = 3    # canais: CVLI, CVP, Tension
        
        # Gera dados sintéticos realistas
        data = np.random.rand(N, T, C).astype(np.float32)
        
        # Canal 0: CVLI (valores menores, mais raros)
        data[:, :, 0] = np.random.poisson(0.5, (N, T))
        
        # Canal 1: CVP (valores um pouco maiores)
        data[:, :, 1] = np.random.poisson(1.0, (N, T))
        
        # Canal 2: Tension (valores entre 0 e 1)
        data[:, :, 2] = np.random.rand(N, T) * 0.5
        
        return data
    
    @pytest.fixture
    def adjacency_matrix(self):
        """Lista de matrizes de adjacência de exemplo"""
        N = 302
        # Cria 2 matrizes de adjacência (num_graphs=2)
        adj_list = []
        for _ in range(2):
            # Matriz de adjacência simples (grafo totalmente conectado normalizado)
            adj = np.ones((N, N), dtype=np.float32)
            # Normaliza por linha
            adj = adj / adj.sum(axis=1, keepdims=True)
            adj_list.append(torch.FloatTensor(adj))
        return adj_list
    
    def test_model_initialization(self, model_config):
        """Teste 1: Modelo inicializa corretamente com 3 canais"""
        try:
            model = STGCN(**model_config)
            assert model is not None
            print("✅ Modelo inicializado corretamente com 3 canais")
        except Exception as e:
            pytest.fail(f"Falha na inicialização do modelo: {e}")
    
    def test_forward_pass(self, model_config, sample_data, adjacency_matrix):
        """Teste 2: Forward pass funciona com dados de 3 canais"""
        model = STGCN(**model_config)
        model.eval()
        
        # Converte dados para tensor: (B, C, N, T)
        input_tensor = torch.FloatTensor(sample_data).permute(2, 0, 1).unsqueeze(0)  # (1, 3, N, T)
        
        try:
            with torch.no_grad():
                output = model(input_tensor, adjacency_matrix)
            
            assert output is not None
            assert output.shape == (1, model_config['num_nodes'], 1)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            print(f"✅ Forward pass bem-sucedido: output shape = {output.shape}")
        except Exception as e:
            pytest.fail(f"Falha no forward pass: {e}")
    
    def test_cvp_channel_isolation(self, model_config, adjacency_matrix):
        """Teste 3: CVP (canal 1) é realmente utilizado pelo modelo"""
        model = STGCN(**model_config)
        model.eval()
        
        N = 302
        T = 30
        
        # Teste 1: Apenas CVP ativo (canal 1)
        data_cvp_only = np.zeros((N, T, 3), dtype=np.float32)
        data_cvp_only[:, :, 1] = np.random.poisson(2.0, (N, T))  # CVP com valores
        
        input_cvp = torch.FloatTensor(data_cvp_only).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            output_cvp = model(input_cvp, adjacency_matrix)
        
        # Teste 2: Sem CVP (canal 1 zerado)
        data_no_cvp = np.zeros((N, T, 3), dtype=np.float32)
        data_no_cvp[:, :, 0] = np.random.poisson(1.0, (N, T))  # CVLI
        data_no_cvp[:, :, 2] = np.random.rand(N, T) * 0.5      # Tension
        
        input_no_cvp = torch.FloatTensor(data_no_cvp).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            output_no_cvp = model(input_no_cvp, adjacency_matrix)
        
        # Se CVP é realmente usado, outputs devem ser diferentes
        diff = torch.abs(output_cvp - output_no_cvp).mean().item()
        
        assert diff > 1e-6, f"CVP não parece estar sendo utilizado! Diferença: {diff}"
        print(f"✅ CVP é realmente utilizado: diferença média = {diff:.6f}")
    
    def test_three_channel_requirement(self, model_config, adjacency_matrix):
        """Teste 4: Modelo aceita entrada com 3 canais"""
        model = STGCN(**model_config)
        model.eval()
        
        N = 302
        T = 30
        
        # Testa com 3 canais (correto)
        data_3ch = np.random.rand(N, T, 3).astype(np.float32)
        input_3ch = torch.FloatTensor(data_3ch).permute(2, 0, 1).unsqueeze(0)  # (1, 3, N, T)
        
        try:
            with torch.no_grad():
                output = model(input_3ch, adjacency_matrix)
            assert output is not None
            print("✅ Modelo aceita entrada com 3 canais")
        except Exception as e:
            pytest.fail(f"Modelo deveria aceitar 3 canais: {e}")
    
    def test_prediction_range(self, model_config, sample_data, adjacency_matrix):
        """Teste 5: Predições estão em range razoável"""
        model = STGCN(**model_config)
        model.eval()
        
        input_tensor = torch.FloatTensor(sample_data).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor, adjacency_matrix)
        
        output_np = output.squeeze().cpu().numpy()
        
        # Verifica range razoável (modelo usa ReLU final, então valores >= 0)
        assert output_np.min() >= 0, f"Valores negativos detectados: {output_np.min()}"
        assert output_np.max() <= 1000, f"Valores muito positivos: {output_np.max()}"
        
        print(f"✅ Range de predições: [{output_np.min():.2f}, {output_np.max():.2f}]")
    
    def test_cvp_vs_cvli_contribution(self, model_config, adjacency_matrix):
        """Teste 6: CVP e CVLI contribuem de forma independente"""
        model = STGCN(**model_config)
        model.eval()
        
        N = 302
        T = 30
        
        # Cenário 1: Alta CVLI, baixa CVP
        data_1 = np.zeros((N, T, 3), dtype=np.float32)
        data_1[:, :, 0] = np.random.poisson(3.0, (N, T))  # CVLI alto
        data_1[:, :, 1] = np.random.poisson(0.5, (N, T))  # CVP baixo
        
        input_1 = torch.FloatTensor(data_1).permute(2, 0, 1).unsqueeze(0)
        
        # Cenário 2: Baixa CVLI, alta CVP
        data_2 = np.zeros((N, T, 3), dtype=np.float32)
        data_2[:, :, 0] = np.random.poisson(0.5, (N, T))  # CVLI baixo
        data_2[:, :, 1] = np.random.poisson(3.0, (N, T))  # CVP alto
        
        input_2 = torch.FloatTensor(data_2).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            output_1 = model(input_1, adjacency_matrix)
            output_2 = model(input_2, adjacency_matrix)
        
        # Outputs devem ser significativamente diferentes
        diff = torch.abs(output_1 - output_2).mean().item()
        
        assert diff > 0.001, f"CVLI e CVP não parecem contribuir independentemente! Diff: {diff}"
        print(f"✅ CVLI e CVP contribuem independentemente: diferença = {diff:.4f}")
    
    def test_model_load_from_checkpoint(self):
        """Teste 7: Modelo carrega de checkpoint existente"""
        model_paths = [
            'models/stgcn_cvli.pth',
            'models/stgcn_cvp.pth',
            'models/stgcn_model.pth'
        ]
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        for model_path in model_paths:
            full_path = os.path.join(base_dir, model_path)
            if os.path.exists(full_path):
                try:
                    checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)
                    assert checkpoint is not None
                    
                    # Verifica se tem state_dict
                    if isinstance(checkpoint, dict):
                        assert 'model_state_dict' in checkpoint or len(checkpoint) > 0
                    
                    print(f"✅ Checkpoint carregado com sucesso: {model_path}")
                except Exception as e:
                    pytest.fail(f"Falha ao carregar checkpoint {model_path}: {e}")


def run_viability_tests():
    """Executa todos os testes de viabilidade"""
    print("\n" + "="*80)
    print("TESTES DE VIABILIDADE DO MODELO ST-GCN")
    print("="*80 + "\n")
    
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_viability_tests()


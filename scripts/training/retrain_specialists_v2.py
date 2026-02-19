import pickle
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import time
import logging

# Adicionar raiz ao path para imports
sys.path.append(os.getcwd())
try:
    from src.core.architectures import DeepSTGAT_64, DeepSTGAT_32
except ImportError:
    # Fallback se rodar de dentro da pasta scripts
    sys.path.append(os.path.join(os.getcwd(), 'src', 'core'))
    from architectures import DeepSTGAT_64, DeepSTGAT_32

# Configuração de Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_detailed.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configurações de Treino
EPOCHS = 60
LR = 0.003
GRADIENT_ACCUMULATION_STEPS = 32 # Simula batch size de 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info(f"Dispositivo de Treinamento: {DEVICE}")

def load_processed_data(region_key):
    """Carrega o pickle processado da região."""
    path = f'data/processed/processed_{region_key}.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def train_specialist(region_key, ModelClass):
    region_label = region_key.upper()
    logging.info(f"="*50)
    logging.info(f"⚡ INICIANDO TREINAMENTO: {region_label}")
    logging.info(f"="*50)
    
    start_time_region = time.time()
    try:
        data = load_processed_data(region_key)
    except FileNotFoundError as e:
        logging.error(f"Erro ao carregar dados para {region_label}: {e}")
        return

    features = data['node_features'] # (N, Total_Days, 29)
    # Mover adjacências para GPU uma única vez
    adj_geo = torch.tensor(data['adj_geo'], dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(data['adj_conflict'], dtype=torch.float32).to(DEVICE)
    
    WINDOW = 30
    PREDICT_HORIZON = 7
    
    N, T_total, C = features.shape
    X_list, y_list = [], []
    
    # Normalização Simples
    features_norm = features.copy()
    for c in range(C):
        mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
        features_norm[:, :, c] = (features[:, :, c] - mean) / std

    # Criar Dataset Deslizante com SUAVIZAÇÃO ESPACIAL (Tensão Regional)
    # Objetivo: Ensinar o modelo a prever a "Mancha Criminal" e não apenas o ponto exato.
    # Se Bairro A tem crime, Bairro B (vizinho) recebe 0.3 de risco no target.
    
    # Pre-computar matriz de adjacência densa para propagação (na CPU primeiro)
    adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32) # (N, N)
    
    for t in range(WINDOW, T_total - PREDICT_HORIZON):
        # Input: Janela (N, 30, 29) -> Transpor para (1, 29, N, 30)
        x_window = features_norm[:, t-WINDOW:t, :] 
        x_tensor = torch.tensor(x_window, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        # Target Original: Soma de CVLI (Canal 0) nos próximos 7 dias
        y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32) # (N,)
        
        # --- ENGENHARIA DE TARGET: SUAVIZAÇÃO ESPACIAL ---
        # Propagar 30% do risco para vizinhos diretos (Simula Tensão Regional)
        # y_smoothed = y_raw + 0.3 * (adj * y_raw)
        neighbor_risk = torch.matmul(adj_dense, y_raw)
        y_target = y_raw + (0.3 * neighbor_risk)
        
        # Normalizar target para escala 0-1 (ajuda na convergência do MSE)
        if y_target.max() > 0:
            y_target = y_target / y_target.max()
            
        y_tensor = y_target.unsqueeze(0) # (1, N)
        
        X_list.append(x_tensor)
        y_list.append(y_tensor)
        
    # Split Treino/Teste: Janela de Validação de 4 Meses (aprox 120 dias)
    val_days = 120
    if len(X_list) > val_days:
        train_X = X_list[:-val_days]
        train_y = y_list[:-val_days]
        val_X = X_list[-val_days:]
        val_y = y_list[-val_days:]
    else:
        # Fallback se dataset for pequeno (80/20)
        split = int(len(X_list) * 0.8)
        train_X = X_list[:split]
        train_y = y_list[:split]
        val_X = X_list[split:]
        val_y = y_list[split:]
    
    logging.info(f"Dataset pronto com Suavização Espacial. Treino: {len(train_X)} | Validação: {len(val_X)} (Janela de 4 meses)")
    
    # Inicializar Modelo
    model = ModelClass(num_nodes=N, in_channels=C, time_steps=WINDOW).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Custom Weighted MSE Loss para priorizar Ranking (P@10)
    # Penaliza mais os erros em nós com alto crime real
    def weighted_mse_loss(pred, target):
        # Peso = 1 + log(1 + target) para suavizar o impacto de outliers mas focar no topo
        weights = 1.0 + torch.log1p(target)
        loss = weights * (pred - target) ** 2
        return loss.mean()
        
    criterion = weighted_mse_loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5)
    
    best_p10 = -1.0
    best_loss_at_p10 = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        # Embaralhar índices manualmente para SGD
        indices = np.random.permutation(len(train_X))
        
        steps = 0
        for idx in indices:
            bx = train_X[idx].to(DEVICE)
            by = train_y[idx].to(DEVICE)
            
            # Forward
            pred = model(bx, [adj_geo, adj_conf])
            
            # Usar Loss Ponderada
            loss = criterion(pred.squeeze(), by.squeeze())
            
            # Normalize loss para acumulação
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            steps += 1
            if steps % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Aplicar updates restantes
        if steps % GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        avg_train = epoch_loss / len(train_X)
        
        # Validação
        model.eval()
        val_loss = 0.0
        p5_list = []
        p10_list = []
        
        with torch.no_grad():
            for i in range(len(val_X)):
                vx = val_X[i].to(DEVICE)
                vy = val_y[i].to(DEVICE)
                vpred = model(vx, [adj_geo, adj_conf])
                val_loss += criterion(vpred.squeeze(), vy.squeeze()).item()
                
                # --- CÁLCULO DE MÉTRICAS DE RANKING (P@k) ---
                y_true = vy.squeeze().cpu().numpy()
                y_pred = vpred.squeeze().cpu().numpy()
                
                # Ignorar dias sem crime nenhum (ruído)
                if np.sum(y_true) > 0:
                    # Índices ordenados do maior para o menor
                    top_true = np.argsort(y_true)[::-1]
                    top_pred = np.argsort(y_pred)[::-1]
                    
                    # P@5
                    k = 5
                    hits5 = len(set(top_true[:k]) & set(top_pred[:k]))
                    p5_list.append(hits5 / k)
                    
                    # P@10
                    k = 10
                    hits10 = len(set(top_true[:k]) & set(top_pred[:k]))
                    p10_list.append(hits10 / k)
        
        avg_val = val_loss / len(val_X)
        avg_p5 = np.mean(p5_list) if p5_list else 0.0
        avg_p10 = np.mean(p10_list) if p10_list else 0.0
        
        scheduler.step(avg_val)
        
        # Logging inteligente com label da região
        logging.info(f"[{region_label}] Epoch {epoch+1:03d}/{EPOCHS} | MSE: {avg_val:.4f} | P@5: {avg_p5*100:.1f}% | P@10: {avg_p10*100:.1f}%")
            
        # Lógica de Salvamento: Prioridade Total ao P@10
        # Se P@10 melhorou -> Salva
        # Se P@10 empatou mas Loss melhorou -> Salva
        
        saved = False
        if avg_p10 > best_p10:
            best_p10 = avg_p10
            best_loss_at_p10 = avg_val
            saved = True
            logging.info(f"🏆 [{region_label}] NOVO RECORDE P@10: {avg_p10*100:.1f}% (MSE: {avg_val:.4f})")
        elif avg_p10 == best_p10 and avg_val < best_loss_at_p10:
            best_loss_at_p10 = avg_val
            saved = True
            logging.info(f"🔹 [{region_label}] P@10 Estável ({avg_p10*100:.1f}%), Melhoria no MSE: {avg_val:.4f}")
            
        if saved:
            torch.save({'model_state_dict': model.state_dict()}, f'models/active/{region_key}_model.pth')

    total_time = time.time() - start_time_region
    logging.info(f"✅ {region_label} CONCLUÍDO. Melhor P@10: {best_p10*100:.1f}%. Tempo: {total_time:.1f}s")

def main():
    os.makedirs('models/active', exist_ok=True)
    
    try:
        train_specialist('fortaleza', DeepSTGAT_64)
        train_specialist('rmf', DeepSTGAT_32)
        train_specialist('interior', DeepSTGAT_32)
        
        logging.info("\n🏆 CICLO DE RETREINO FINALIZADO COM SUCESSO.")
        
    except Exception as e:
        logging.error(f"❌ ERRO FATAL NO PROCESSO: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()

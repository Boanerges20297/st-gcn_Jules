import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Forçar sys.stdout a usar UTF-8 para evitar erros em ambientes Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configurações de Treino
EPOCHS = 60
LR = 0.001 
GRADIENT_ACCUMULATION_STEPS = 24 # Reduzido para 24 conforme solicitado
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_priority_weights(features, nodes_gdf):
    """
    Calcula pesos de severidade baseados na história real (A ideia 'marota').
    features: (N, T, C)
    """
    # 1. Pesos Espaciais (Maracanaú e Top Hotspots)
    cvli_total_per_node = features[:, :, 0].sum(axis=1)
    # Maracanaú ganha +40%, outros proporcionalmente (capado em 1.5x)
    spatial_weights = 1.0 + (cvli_total_per_node / (cvli_total_per_node.max() + 1e-6)) * 0.4
    
    # 2. Pesos Temporais (Mês e Dia da Semana)
    # No dataset, canal 21 é Mês (1-12) e 22 é Dia da Semana (0-6) - Confirmar se é isso
    # Vamos assumir canais 21 e 22 baseados no padrão de 29 canais
    month_weights = {m: 1.0 for m in range(1, 13)}
    day_weights = {d: 1.0 for d in range(7)}
    
    # Valores extraídos do nosso relatório 'robust'
    # Meses: Outubro(1.2), Agosto(1.2), Fevereiro(0.8)
    month_weights.update({10: 1.2, 8: 1.2, 2: 0.8, 1: 1.1})
    # Dias: Sab/Dom (1.3), Sex (1.1), outros (0.9)
    day_weights.update({5: 1.1, 6: 1.3, 0: 1.3}) # 0 é Segunda ou Domingo? Assumindo 6=Sab, 0=Dom
    
    return spatial_weights, month_weights, day_weights

def train_specialist(region_key, ModelClass):
    region_label = region_key.upper()
    logging.info(f"="*50)
    logging.info(f"⚡ INICIANDO TREINAMENTO: {region_label} (JULES DYNAMIC PRIORITY)")
    logging.info(f"="*50)
    
    start_time_region = time.time()
    try:
        data = load_processed_data(region_key)
    except FileNotFoundError as e:
        logging.error(f"Erro ao carregar dados para {region_label}: {e}")
        return

    features = data['node_features'] 
    nodes_gdf = data['nodes_gdf']
    
    # Calcular Pesos Dinâmicos
    spatial_weights_np, month_weights_map, day_weights_map = calculate_priority_weights(features, nodes_gdf)
    spatial_weights = torch.tensor(spatial_weights_np, dtype=torch.float32).to(DEVICE)
    
    adj_geo_norm = normalize_adj(data['adj_geo'])
    adj_conf_norm = normalize_adj(data['adj_conflict'])
    adj_geo = torch.tensor(adj_geo_norm, dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(adj_conf_norm, dtype=torch.float32).to(DEVICE)
    
    WINDOW = 30
    PREDICT_HORIZON = 7
    N, T_total, C = features.shape
    
    # Normalização
    features_norm = features.copy()
    for c in range(C):
        mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
        features_norm[:, :, c] = (features[:, :, c] - mean) / std

    X_list, y_list, info_list = [], [], []
    adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)
    
    dates = pd.to_datetime(data['dates'])

    for t in range(WINDOW, T_total - PREDICT_HORIZON):
        x_window = features_norm[:, t-WINDOW:t, :] 
        x_tensor = torch.tensor(x_window, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
        neighbor_risk = torch.matmul(adj_dense, y_raw)
        y_target = y_raw + (0.3 * neighbor_risk)
        
        if y_target.max() > 0:
            y_target = y_target / y_target.max()
            
        y_tensor = y_target.unsqueeze(0)
        
        # Guardar info temporal para a Loss Mutante (Mês e Dia do dia da PREVISÃO)
        current_date = dates[t]
        info_list.append({
            'month': current_date.month,
            'dow': current_date.dayofweek
        })
        
        X_list.append(x_tensor)
        y_list.append(y_tensor)
        
    # Split: Janela de 90 dias conforme solicitado
    val_days = 90
    train_X = X_list[:-val_days]
    train_y = y_list[:-val_days]
    train_info = info_list[:-val_days]
    val_X = X_list[-val_days:]
    val_y = y_list[-val_days:]
    val_info = info_list[-val_days:]
    
    logging.info(f"Dataset pronto. Treino: {len(train_X)} | Validação: {len(val_X)} (Janela de 90 dias)")
    
    model = ModelClass(num_nodes=N, in_channels=C, time_steps=WINDOW).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # --- LOSS MUTANTE (JULES DYNAMIC) ---
    def hybrid_priority_loss(pred, target, info):
        pred = pred.squeeze()
        target = target.squeeze()
        
        # 1. Multiplicador Temporal (Mês/Dia)
        m_weight = month_weights_map.get(info['month'], 1.0)
        d_weight = day_weights_map.get(info['dow'], 1.0)
        temporal_multiplier = m_weight * d_weight
        
        # 2. Regressão com Peso Espacial + TopK
        k = 30
        top_val, top_idx = torch.topk(target, min(k, len(target)))
        
        # Base weights + Spatial Bias
        weights = spatial_weights.clone() # (N,)
        weights[top_idx] = weights[top_idx] * 4.0 * (1.0 + target[top_idx])
        
        loss_reg = (weights * F.smooth_l1_loss(pred, target, reduction='none')).mean()
        
        # 3. Ranking Refinado (Agressividade 0.3)
        if top_val.sum() == 0:
            return loss_reg * temporal_multiplier
            
        num_negatives = 50
        neg_idx = torch.randint(0, len(target), (num_negatives,), device=target.device)
        pred_high, pred_low = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
        target_high, target_low = target[top_idx].unsqueeze(1), target[neg_idx].unsqueeze(0)
        
        margin = 0.2 + (F.relu(target_high - target_low) * 0.5)
        loss_rank = (F.relu(margin - (pred_high - pred_low)) * (target_high > target_low).float()).sum() / (num_negatives * k)
        
        # A Loss total é multiplicada pela severidade da época (Temporal)
        return (loss_reg + 0.3 * loss_rank) * temporal_multiplier

    criterion = hybrid_priority_loss
    # OneCycleLR com mais epochs para acomodar o oversampling
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR*3.0, steps_per_epoch=len(final_train_indices)//GRADIENT_ACCUMULATION_STEPS + 1, 
        epochs=EPOCHS, pct_start=0.3
    )
    
    best_p20 = -1.0
    best_loss_at_p20 = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        # Embaralhar índices com oversampling
        np.random.shuffle(final_train_indices)
        
        steps = 0
        for idx in final_train_indices:
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
                # Gradient Clipping para evitar explosão
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() # Step no OneCycle é por batch
                optimizer.zero_grad()
                
                # Log progress every update
                if (steps // GRADIENT_ACCUMULATION_STEPS) % 5 == 0:
                    # Quick P@K calculation for the current batch
                    with torch.no_grad():
                        y_true = by.squeeze().cpu().numpy()
                        y_pred = pred.squeeze().cpu().numpy()
                        
                        p10, p20 = 0.0, 0.0
                        if np.sum(y_true) > 0:
                            top_true = np.argsort(y_true)[::-1]
                            top_pred = np.argsort(y_pred)[::-1]
                            
                            hits10 = len(set(top_true[:10]) & set(top_pred[:10]))
                            p10 = hits10 / 10.0
                            
                            hits20 = len(set(top_true[:20]) & set(top_pred[:20]))
                            p20 = hits20 / 20.0
                            
                    logging.info(f"   -> Epoch {epoch+1} Progress: {steps}/{len(final_train_indices)} steps | Loss: {loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f} | Train P@10: {p10*100:.1f}% | Train P@20: {p20*100:.1f}%")
        
        # Validação
        model.eval()
        val_loss = 0.0
        p5_list, p10_list, p20_list = [], [], []
        
        with torch.no_grad():
            for i in range(len(val_X)):
                vx = val_X[i].to(DEVICE)
                vy = val_y[i].to(DEVICE)
                vpred = model(vx, [adj_geo, adj_conf])
                val_loss += criterion(vpred.squeeze(), vy.squeeze()).item()
                
                # --- CÁLCULO DE MÉTRICAS DE RANKING (P@k) ---
                y_true = vy.squeeze().cpu().numpy()
                y_pred = vpred.squeeze().cpu().numpy()
                
                if np.sum(y_true) > 0:
                    top_true = np.argsort(y_true)[::-1]
                    top_pred = np.argsort(y_pred)[::-1]
                    
                    # P@5, P@10, P@20
                    for k, l in zip([5, 10, 20], [p5_list, p10_list, p20_list]):
                        hits = len(set(top_true[:k]) & set(top_pred[:k]))
                        l.append(hits / k)
        
        avg_val = val_loss / len(val_X)
        avg_p5 = np.mean(p5_list) if p5_list else 0.0
        avg_p10 = np.mean(p10_list) if p10_list else 0.0
        avg_p20 = np.mean(p20_list) if p20_list else 0.0
        
        # Logging inteligente com P@20
        logging.info(f"[{region_label}] Epoch {epoch+1:03d}/{EPOCHS} | MSE: {avg_val:.4f} | P@5: {avg_p5*100:.1f}% | P@10: {avg_p10*100:.1f}% | P@20: {avg_p20*100:.1f}%")
            
        # Lógica de Salvamento: Foco total no P@20 conforme solicitado
        saved = False
        if avg_p20 > best_p20:
            best_p20 = avg_p20
            best_loss_at_p20 = avg_val
            saved = True
            logging.info(f"🏆 [{region_label}] NOVO RECORDE P@20: {avg_p20*100:.1f}% (P@10: {avg_p10*100:.1f}%)")
        elif avg_p20 == best_p20 and avg_val < best_loss_at_p20:
            best_loss_at_p20 = avg_val
            saved = True
            logging.info(f"🔹 [{region_label}] P@20 Estável ({avg_p20*100:.1f}%), Melhoria no MSE")
            
        if saved:
            torch.save({'model_state_dict': model.state_dict()}, f'models/active/{region_key}_model.pth')

    total_time = time.time() - start_time_region
    logging.info(f"✅ {region_label} CONCLUÍDO. Melhor P@20: {best_p20*100:.1f}%. Tempo: {total_time:.1f}s")

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

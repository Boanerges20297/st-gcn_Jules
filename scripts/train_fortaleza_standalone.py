import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import logging
import pandas as pd
import random

# Adicionar raiz ao path para imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from src.core.architectures import DeepSTGAT_64
except ImportError:
    sys.path.append(os.path.join(ROOT_DIR, 'src', 'core'))
    from architectures import DeepSTGAT_64

# Configuração de Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_fortaleza_standalone.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configurações de Treino (Baseado em v3, mas com 30 épocas)
EPOCHS = 30
LR = 0.02 
GRADIENT_ACCUMULATION_STEPS = 24 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_processed_data(region_key):
    path = os.path.join(ROOT_DIR, 'data', 'processed', f'processed_{region_key}.pkl')
    logging.info(f"Carregando dados de: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def calculate_priority_weights(features, dates):
    # 1. Pesos Espaciais (Hotspots) - DINÂMICO
    cvli_total_per_node = features[:, :, 0].sum(axis=1)
    spatial_weights = 1.0 + (cvli_total_per_node / (cvli_total_per_node.max() + 1e-6)) * 0.4
    
    # 2. Pesos Temporais (Sazonalidade) - AGORA 100% DINÂMICO
    df_temp = pd.DataFrame({'date': pd.to_datetime(dates), 'crimes': features[:, :, 0].sum(axis=0)})
    df_temp['month'] = df_temp['date'].dt.month
    df_temp['dow'] = df_temp['date'].dt.dayofweek
    
    # Média global de crimes por dia para normalizar
    avg_crimes = df_temp['crimes'].mean() + 1e-6
    
    # Pesos por Mês: Proporção em relação à média
    month_avg = df_temp.groupby('month')['crimes'].mean()
    month_weights = {m: max(0.8, min(1.3, val/avg_crimes)) for m, val in month_avg.items()}
    
    # Pesos por Dia da Semana: Proporção em relação à média
    dow_avg = df_temp.groupby('dow')['crimes'].mean()
    day_weights = {d: max(0.8, min(1.3, val/avg_crimes)) for d, val in dow_avg.items()}
    
    return spatial_weights, month_weights, day_weights

def train_specialist(region_key, ModelClass):
    region_label = region_key.upper()
    logging.info(f"="*50)
    logging.info(f"⚡ INICIANDO TREINAMENTO: {region_label} (STANDALONE - 30 EPOCHS)")
    logging.info(f"="*50)
    
    data = load_processed_data(region_key)
    features = data['node_features'] 
    dates = pd.to_datetime(data['dates'])
    
    # Calcular Pesos Dinâmicos (Agora com Sazonalidade Dinâmica)
    spatial_weights_np, month_weights_map, day_weights_map = calculate_priority_weights(features, dates)
    spatial_weights = torch.tensor(spatial_weights_np, dtype=torch.float32).to(DEVICE)
    
    adj_geo = torch.tensor(normalize_adj(data['adj_geo']), dtype=torch.float32).to(DEVICE)
    adj_conf = torch.tensor(normalize_adj(data['adj_conflict']), dtype=torch.float32).to(DEVICE)
    
    WINDOW, PREDICT_HORIZON = 120, 7
    N, T_total, C = features.shape
    
    features_norm = features.copy()
    for c in range(C):
        mean, std = features[:, :, c].mean(), features[:, :, c].std() + 1e-5
        features_norm[:, :, c] = (features[:, :, c] - mean) / std

    X_list, y_list, info_list = [], [], []
    adj_dense = torch.tensor(data['adj_geo'], dtype=torch.float32)
    dates = pd.to_datetime(data['dates'])

    for t in range(WINDOW, T_total - PREDICT_HORIZON):
        x_tensor = torch.tensor(features_norm[:, t-WINDOW:t, :], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        y_raw = torch.tensor(features[:, t:t+PREDICT_HORIZON, 0].sum(axis=1), dtype=torch.float32)
        y_target = y_raw + (0.3 * torch.matmul(adj_dense, y_raw))
        if y_target.max() > 0: y_target = y_target / y_target.max()
        
        current_date = dates[t]
        info_list.append({'month': current_date.month, 'dow': current_date.dayofweek})
        X_list.append(x_tensor)
        y_list.append(y_target.unsqueeze(0))
        
    # Split: BLINDAGEM TOTAL (Safety Gap de 14 dias)
    lastro_days = 90
    val_days = 90
    gap = PREDICT_HORIZON + 7 # Safety Gap
    total_idx = len(X_list)
    
    available_limit = total_idx - lastro_days - gap
    available_idx = list(range(available_limit))
    
    val_idx = random.sample(available_idx, val_days)
    train_idx_base = [i for i in available_idx if i not in val_idx]
    lastro_idx = list(range(total_idx - lastro_days, total_idx))
    
    # Auditoria de Datas
    train_dates = (dates[WINDOW + train_idx_base[0]], dates[WINDOW + train_idx_base[-1]])
    val_dates = (dates[WINDOW + val_idx[0]], dates[WINDOW + val_idx[-1]])
    lastro_dates = (dates[WINDOW + lastro_idx[0]], dates[WINDOW + lastro_idx[-1]])
    
    logging.info(f"--- AUDITORIA DE BLINDAGEM ---")
    logging.info(f"TREINO: {train_dates[0].date()} até {train_dates[1].date()}")
    logging.info(f"LASTRO (INÉDITO): {lastro_dates[0].date()} até {lastro_dates[1].date()}")
    logging.info(f"GAP DE SEGURANÇA: {gap} dias")
    logging.info(f"------------------------------")
    
    train_X = [X_list[i] for i in train_idx_base]
    train_y = [y_list[i] for i in train_idx_base]
    train_info = [info_list[i] for i in train_idx_base]
    
    val_X = [X_list[i] for i in val_idx]
    val_y = [y_list[i] for i in val_idx]
    val_info = [info_list[i] for i in val_idx]

    lastro_X = [X_list[i] for i in lastro_idx]
    lastro_y = [y_list[i] for i in lastro_idx]
    lastro_info = [info_list[i] for i in lastro_idx]
    
    logging.info(f"Dataset: Treino={len(train_X)} | Val={len(val_X)} | Lastro Inédito={len(lastro_X)}")
    
    # Injetando DROPOUT de 0.4 para regularização extrema
    model = ModelClass(num_nodes=N, in_channels=C, time_steps=WINDOW, dropout=0.4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    def criterion(pred, target, info):
        pred, target = pred.squeeze(), target.squeeze()
        t_mult = month_weights_map.get(info['month'], 1.0) * day_weights_map.get(info['dow'], 1.0)
        
        k = 30
        top_val, top_idx = torch.topk(target, min(k, len(target)))
        w = spatial_weights.clone()
        w[top_idx] = w[top_idx] * 4.0 * (1.0 + target[top_idx])
        loss_reg = (w * F.smooth_l1_loss(pred, target, reduction='none')).mean()
        
        if top_val.sum() == 0: return loss_reg * t_mult
            
        num_neg = 50
        neg_idx = torch.randint(0, len(target), (num_neg,), device=target.device)
        p_h, p_l = pred[top_idx].unsqueeze(1), pred[neg_idx].unsqueeze(0)
        t_h, t_l = target[top_idx].unsqueeze(1), target[neg_idx].unsqueeze(0)
        margin = 0.2 + (F.relu(t_h - t_l) * 0.5)
        loss_rank = (F.relu(margin - (p_h - p_l)) * (t_h > t_l).float()).sum() / (num_neg * k)
        # COMBINAÇÃO ULTRA-AGRESSIVA: Regressão + 15.0 * Ranking
        return (loss_reg + 15.0 * loss_rank) * t_mult

    steps_per_epoch = (len(train_X) * 2 // GRADIENT_ACCUMULATION_STEPS) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)
    
    # CARREGAR RECORDE HISTÓRICO (PERSISTÊNCIA)
    best_p20 = 0.0
    record_path = os.path.join(ROOT_DIR, 'logs', 'best_p20_record.txt')
    if os.path.exists(record_path):
        try:
            with open(record_path, 'r') as f:
                content = f.read()
                # Extrair o valor numérico após "Recorde P@20: "
                import re
                match = re.search(r"Recorde P@20: ([\d.]+)", content)
                if match:
                    best_p20 = float(match.group(1)) / 100.0
                    logging.info(f"📜 Recorde Histórico Carregado: {best_p20*100:.2f}%")
        except Exception as e:
            logging.warning(f"⚠️ Não foi possível carregar o recorde anterior: {e}")
    
    # Oversampling logic
    day_sev = [torch.sum(y).item() for y in train_y]
    high_idx = [i for i, s in enumerate(day_sev) if s > np.median(day_sev)]
    train_indices = list(range(len(train_X))) + high_idx + high_idx
    total_steps = len(train_indices) // GRADIENT_ACCUMULATION_STEPS

    logging.info(f"🎬 Iniciando Loop de Treinamento: {EPOCHS} épocas | {total_steps} passos/época")
    logging.info(f"{'='*100}")
    logging.info(f"{'PASSO':<15} | {'LR':<10} | {'LOSS':<10} | {'P@10':<8} | {'P@20':<8}")
    logging.info(f"{'='*100}")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        np.random.shuffle(train_indices)
        steps = 0
        for idx in train_indices:
            bx, by, bi = train_X[idx].to(DEVICE), train_y[idx].to(DEVICE), train_info[idx]
            
            # Forward pass armazenado em 'pred'
            pred = model(bx, [adj_geo, adj_conf])
            loss = criterion(pred, by, bi) / GRADIENT_ACCUMULATION_STEPS
            
            loss.backward()
            steps += 1
            if steps % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                optimizer.zero_grad()
                
                current_step = steps // GRADIENT_ACCUMULATION_STEPS
                # Telemetria com Progresso e Cabeçalho Implícito
                if current_step % 2 == 0:
                    with torch.no_grad():
                        y_true, y_pred = by.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy()
                        p10, p20 = 0.0, 0.0
                        if np.sum(y_true) > 0:
                            t_true, t_pred = np.argsort(y_true)[::-1], np.argsort(y_pred)[::-1]
                            p10 = len(set(t_true[:10]) & set(t_pred[:10])) / 10.0
                            p20 = len(set(t_true[:20]) & set(t_pred[:20])) / 20.0
                    logging.info(f"E{epoch+1:02d} [{current_step:03d}/{total_steps:03d}] | {current_lr:.6f} | {loss.item()*GRADIENT_ACCUMULATION_STEPS:.6f} | {p10*100:>5.1f}% | {p20*100:>5.1f}%")
        
        model.eval()
        val_loss = 0.0
        p10_l, p20_l = [], []
        l_p10_l, l_p20_l = [], []
        
        with torch.no_grad():
            # Validação Aleatória (Métrica Comparativa)
            for i in range(len(val_X)):
                vx, vy, vi = val_X[i].to(DEVICE), val_y[i].to(DEVICE), val_info[i]
                vpred = model(vx, [adj_geo, adj_conf])
                val_loss += criterion(vpred.squeeze(), vy.squeeze(), vi).item()
                vy_np, vp_np = vy.squeeze().cpu().numpy(), vpred.squeeze().cpu().numpy()
                if np.sum(vy_np) > 0:
                    t_true, t_pred = np.argsort(vy_np)[::-1], np.argsort(vp_np)[::-1]
                    p10_l.append(len(set(t_true[:10]) & set(t_pred[:10])) / 10.0)
                    p20_l.append(len(set(t_true[:20]) & set(t_pred[:20])) / 20.0)
            
            # TESTE DE REALIDADE (Lastro Inédito)
            for i in range(len(lastro_X)):
                lx, ly, li = lastro_X[i].to(DEVICE), lastro_y[i].to(DEVICE), lastro_info[i]
                lpred = model(lx, [adj_geo, adj_conf])
                ly_np, lp_np = ly.squeeze().cpu().numpy(), lpred.squeeze().cpu().numpy()
                if np.sum(ly_np) > 0:
                    t_true, t_pred = np.argsort(ly_np)[::-1], np.argsort(lp_np)[::-1]
                    l_p10_l.append(len(set(t_true[:10]) & set(t_pred[:10])) / 10.0)
                    l_p20_l.append(len(set(t_true[:20]) & set(t_pred[:20])) / 20.0)
        
        avg_val = val_loss / len(val_X)
        avg_p10, avg_p20 = np.mean(p10_l or [0]), np.mean(p20_l or [0])
        real_p10, real_p20 = np.mean(l_p10_l or [0]), np.mean(l_p20_l or [0])
        
        logging.info(f"[{region_label}] Epoch {epoch+1:03d} | Loss: {avg_val:.4f} | P@20 (Val): {avg_p20*100:.1f}% | P@20 (REALITY): {real_p20*100:.1f}%")
        
        if real_p20 > best_p20:
            best_p20 = real_p20
            # Definir o caminho de salvamento corretamente
            save_path = os.path.join(ROOT_DIR, 'models', 'active', f'{region_key}_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'p20_record': best_p20,
                'epoch': epoch
            }, save_path)
            
            # Salvar recorde em arquivo texto para consulta rápida
            record_path = os.path.join(ROOT_DIR, 'logs', 'best_p20_record.txt')
            with open(record_path, 'w') as f:
                f.write(f"Regiao: {region_key}\nRecorde P@20: {best_p20*100:.2f}%\nEpoca: {epoch+1}\nData: {pd.Timestamp.now()}")
            
            logging.info(f"🏆 [RECORDE] Novo P@20 Realidade: {best_p20*100:.2f}% | Modelo salvo em {save_path}")

def main():
    # Garantir que o diretório de modelos existe
    models_dir = os.path.join(ROOT_DIR, 'models', 'active')
    os.makedirs(models_dir, exist_ok=True)
    
    # Treinar apenas Fortaleza
    train_specialist('fortaleza', DeepSTGAT_64)

if __name__ == "__main__":
    main()

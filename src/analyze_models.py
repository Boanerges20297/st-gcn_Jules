import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# Adjust path to include src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import STGCN

class STGCNAnalyzer(STGCN):
    """
    Inherits from STGCN but overrides forward to return intermediate layer outputs
    for analysis of over-smoothing.
    """
    def __init__(self, num_nodes, in_channels, time_steps, num_classes=2):
        super(STGCNAnalyzer, self).__init__(num_nodes, in_channels, time_steps, num_classes)

    def forward(self, x, adj):
        # Layer 1
        x1 = self.layer1(x, adj)
        # Layer 2
        x2 = self.layer2(x1, adj)
        # Final layers
        out = self.conv_final(x2)
        out = out.squeeze(-1).permute(0, 2, 1)
        out = self.fc(out)

        return out, x1, x2

def calculate_mad(tensor_data):
    """
    Calculates Mean Average Distance (MAD) for a batch of node features.
    MAD measures the average cosine distance between all pairs of nodes.
    Lower MAD indicates over-smoothing (nodes becoming indistinguishable).

    tensor_data shape: [Batch, Time, Nodes, Channels] or similar.
    We need to average over Batch and Time to get a single vector per Node.
    Actually, over-smoothing is spatial, so we check diversity across Nodes.

    We'll treat (Batch * Time) as samples.
    """
    # If data is 4D: [B, T, N, C] -> flatten to [B*T, N, C]
    if tensor_data.dim() == 4:
        B, T, N, C = tensor_data.size()
        data = tensor_data.reshape(-1, N, C) # [Samples, Nodes, Features]
    elif tensor_data.dim() == 3:
        # [B, N, C]
        data = tensor_data
    else:
        return 0.0

    # Calculate cosine distance between nodes for each sample

    mad_values = []
    # To save memory, strictly limit samples
    num_samples = data.size(0)
    # Reduce sample size to max 5 to avoid OOM during training loop
    sample_size = min(5, num_samples)
    indices = np.random.choice(num_samples, size=sample_size, replace=False)

    subset = data[indices] # [5, N, C]

    # Process one by one to save memory
    for i in range(subset.size(0)):
        node_feats = subset[i] # [N, C]
        # print(f"DEBUG: calculating MAD for N={node_feats.size(0)}", flush=True)

        # Avoid division by zero
        norm = node_feats.norm(p=2, dim=1, keepdim=True)
        norm[norm == 0] = 1e-8
        node_feats_norm = node_feats / norm

        # Cosine similarity matrix: [N, N]
        sim_matrix = torch.mm(node_feats_norm, node_feats_norm.t())

        # Cosine distance = 1 - similarity
        dist_matrix = 1 - sim_matrix

        # Average distance (exclude diagonal which is 0)
        N_nodes = node_feats.size(0)
        if N_nodes > 1:
            avg_dist = dist_matrix.sum() / (N_nodes * N_nodes)
            mad_values.append(avg_dist.item())

        del sim_matrix, dist_matrix, node_feats_norm

    return np.mean(mad_values) if mad_values else 0.0

def prepare_dataset(node_features, history_window, horizon, feature_idx):
    """
    Prepares dataset similar to train_separate.py
    """
    num_nodes, num_timesteps, num_features = node_features.shape
    features = node_features[:, :, feature_idx:feature_idx+1] # Keep dims [N, T, 1]

    X, Y = [], []
    valid_range = num_timesteps - history_window - horizon + 1

    # Limit total samples to avoid OOM on limited environments
    # Take latest 32 samples max for speed
    start_idx = max(0, valid_range - 32)

    print(f"Preparing dataset: Window={history_window}, Horizon={horizon}, Range=[{start_idx}:{valid_range}]")

    for i in range(start_idx, valid_range):
        # Input: [N, Window, 1]
        window = features[:, i:i+history_window, :]
        # Target: Sum over horizon [N, 1]
        target = np.sum(features[:, i+history_window:i+history_window+horizon, :], axis=1)

        # Transpose X to [Channels, Nodes, Time] -> [1, N, Window]
        X.append(np.transpose(window, (2, 0, 1)).astype(np.float32))
        Y.append(target.astype(np.float32))

    return np.array(X), np.array(Y)

def weighted_mse_loss(input, target):
    return torch.mean((input - target) ** 2)

def run_analysis_for_task(task_name, feature_idx, history_window, horizon, epochs=5, batch_size=8):
    print(f"--- Running Analysis for {task_name} (Window={history_window}, Horizon={horizon}, Batch={batch_size}) ---")

    # 1. Load Data
    data_file = os.path.join(ROOT, 'data', 'processed', 'processed_graph_data.pkl')
    with open(data_file, 'rb') as f:
        pack = pickle.load(f)

    node_features = pack['node_features']
    adj = pack['adj_matrix']
    dates = pack.get('dates')

    # Filter dates >= 2024-01-01 if needed (matching train_separate.py logic)
    if dates is not None:
        dates = list(dates)
        start_idx = 0
        for idx, d in enumerate(dates):
            if str(d) >= '2024-01-01':
                start_idx = idx
                break
        if start_idx > 0:
            node_features = node_features[:, start_idx:, :]

    # REDUCE NODES for analysis speed
    # Only use first 200 nodes
    node_features = node_features[:200, :, :]
    adj = adj[:200, :200]

    # 2. Prepare Dataset
    X, Y = prepare_dataset(node_features, history_window, horizon, feature_idx)

    if len(X) == 0:
        print(f"Not enough data for {task_name}")
        return

    # Split Train/Val (80/20) - Chronological Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # 3. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare normalized Adjacency
    adj_t = torch.FloatTensor(adj)
    rowsum = adj_t.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_t), d_mat_inv_sqrt).to(device)

    num_nodes = node_features.shape[0]
    print(f"Number of nodes: {num_nodes}")
    model = STGCNAnalyzer(num_nodes=num_nodes, in_channels=1, time_steps=history_window, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Metrics storage
    history = {
        'train_loss': [],
        'val_loss': [],
        'mad_input': [],
        'mad_layer1': [],
        'mad_layer2': []
    }

    # Store predictions for scatter plot (last epoch)
    final_preds = []
    final_targets = []

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        batch_losses = []

        # Monitor MAD during training
        mad_in_list = []
        mad_l1_list = []
        mad_l2_list = []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            out, l1, l2 = model(bx, norm_adj)
            loss = weighted_mse_loss(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_losses.append(loss.item())

            # Calculate MAD for one batch (only the first one to save time)
            if len(mad_in_list) < 1:
                # bx shape: [B, C, N, T] -> Permute to [B, T, N, C] for consistent MAD calc
                bx_p = bx.permute(0, 3, 2, 1)
                mad_in_list.append(calculate_mad(bx_p))

                # l1 shape: [B, T, N, C] (from STGCNLayer output)
                mad_l1_list.append(calculate_mad(l1))

                # l2 shape: [B, T, N, C]
                mad_l2_list.append(calculate_mad(l2))

        avg_train_loss = np.mean(batch_losses)
        avg_mad_in = np.mean(mad_in_list)
        avg_mad_l1 = np.mean(mad_l1_list)
        avg_mad_l2 = np.mean(mad_l2_list)

        # Validation
        model.eval()
        val_losses = []
        if epoch == epochs:
            final_preds = []
            final_targets = []

        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                out, _, _ = model(vx, norm_adj)
                loss = weighted_mse_loss(out, vy)
                val_losses.append(loss.item())

                if epoch == epochs:
                    final_preds.append(out.cpu().numpy())
                    final_targets.append(vy.cpu().numpy())

        avg_val_loss = np.mean(val_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mad_input'].append(avg_mad_in)
        history['mad_layer1'].append(avg_mad_l1)
        history['mad_layer2'].append(avg_mad_l2)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | MAD L2: {avg_mad_l2:.4f}")

    # 5. Generate Plots
    plot_loss(history, task_name)
    plot_smoothing(history, task_name)

    if final_preds:
        all_preds = np.concatenate(final_preds, axis=0).flatten()
        all_targets = np.concatenate(final_targets, axis=0).flatten()
        plot_scatter(all_targets, all_preds, task_name)

    return history

def plot_loss(history, task_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curves - {task_name}')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/loss_curves_{task_name}.png')
    plt.close()

def plot_smoothing(history, task_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history['mad_input'], label='Input MAD', linestyle='--')
    plt.plot(history['mad_layer1'], label='Layer 1 MAD')
    plt.plot(history['mad_layer2'], label='Layer 2 MAD')
    plt.title(f'Over-smoothing Analysis (MAD) - {task_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Average Distance (Diversity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/oversmoothing_{task_name}.png')
    plt.close()

def plot_scatter(y_true, y_pred, task_name):
    plt.figure(figsize=(8, 8))

    # Sample if too many points
    if len(y_true) > 5000:
        idx = np.random.choice(len(y_true), 5000, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(f'Prediction vs Ground Truth - {task_name}')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/predictions_{task_name}.png')
    plt.close()

if __name__ == "__main__":
    # Ensure plots dir exists
    os.makedirs('plots', exist_ok=True)

    # CVLI Analysis
    # Reduced window to 30 for analysis speed (demonstration purpose)
    run_analysis_for_task('CVLI', feature_idx=0, history_window=30, horizon=3, epochs=20, batch_size=8)

    # CVP Analysis
    run_analysis_for_task('CVP', feature_idx=1, history_window=30, horizon=1, epochs=20, batch_size=8)

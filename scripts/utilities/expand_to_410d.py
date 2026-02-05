import pickle
import numpy as np
from src.llm_service import get_semantic_embeddings_batch
from src.ranking_features import expand_features_with_semantics
import logging

logging.basicConfig(level=logging.INFO)

# Load data
print('Loading data...')
with open('data/processed/processed_graph_data.pkl', 'rb') as f:
    data = pickle.load(f)

node_features = data['node_features']
print(f'Original shape: {node_features.shape}')

# Get node IDs and bairro names (mock for now)
node_ids = list(range(node_features.shape[0]))
# Simplified mapping - in real case would come from data
bairro_names = {i: f'Bairro_{i%31}' for i in node_ids}
bairro_mapping = {
    0: 'Aldeota', 1: 'Barroso', 2: 'Parangaba', 3: 'Meireles', 4: 'Praia de Iracema',
    5: 'Joquei Clube', 6: 'Lagoa', 7: 'Varjota', 8: 'Centro', 9: 'Mucuripe',
    10: 'Cocó', 11: 'Salinas', 12: 'Pirambu', 13: 'Bom Meigo', 14: 'Carlito Pamplona',
    15: 'Henrique Jorge', 16: 'Pedra Mole', 17: 'Barrinha', 18: 'Taboca', 19: 'Pici',
    20: 'Antônio Bezerra', 21: 'Damas', 22: 'Quintino Cunha', 23: 'Siqueira', 24: 'Castelão',
    25: 'Vila União', 26: 'Granja Portugal', 27: 'Vila Velha', 28: 'Sabiazinho', 29: 'Messejana', 30: 'Ancuri'
}
bairro_names = {i: bairro_mapping[i % 31] for i in node_ids}

# Generate embeddings
print('Generating semantic embeddings...')
unique_bairros = list(set(bairro_names.values()))
print(f'Unique bairros: {len(unique_bairros)}')

embeddings = get_semantic_embeddings_batch(unique_bairros)
print(f'Generated {len(embeddings)} embeddings')
print(f'First embedding shape: {np.array(embeddings[list(embeddings.keys())[0]]).shape}')

# Expand features
print('Expanding features...')
features_410d = expand_features_with_semantics(node_features, node_ids, bairro_names)
print(f'Expanded shape: {features_410d.shape}')

# Save
output_file = 'data/processed/processed_graph_data_semantic.pkl'
print(f'Saving to {output_file}...')
data['node_features'] = features_410d
with open(output_file, 'wb') as f:
    pickle.dump(data, f)

print('✓ Done!')

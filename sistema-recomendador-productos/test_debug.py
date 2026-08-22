import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from gensim.models import Word2Vec
import pandas as pd
from pathlib import Path

# Load
art = joblib.load(Path('models/hibrido_v2.pkl'))
data = np.load(Path('models/embeddings_word2vec.npz'), allow_pickle=True)
embeddings = data['embeddings']
product_ids = data['product_ids']
df_prod = pd.read_csv(Path('data/productos.csv'))
df_int = pd.read_csv(Path('data/interacciones.csv'))

uid = 'u0007'
usuario_idx = art['idx_usuario'][uid]
print(f'Usuario idx: {usuario_idx}')

# Get product IDs from history
hist_pids = df_int[df_int['user_id'] == uid]['product_id'].values[:5]
print('Historial PIDs:', hist_pids)

# Map to embedding indices
# product_ids is the array of product_ids used when creating embeddings
print('Product IDs sample:', product_ids[:5])

# Find indices in embeddings
hist_emb_idx = []
for pid in hist_pids:
    idx = np.where(product_ids == pid)[0]
    if len(idx) > 0:
        hist_emb_idx.append(idx[0])
        print(f'  {pid} -> embedding idx {idx[0]}')
    else:
        print(f'  {pid} -> NOT FOUND')

print('Historial embedding indices:', hist_emb_idx)

# Calculate similarity
emb_historial = embeddings[hist_emb_idx].mean(axis=0)
sims = cosine_similarity(emb_historial.reshape(1, -1), embeddings).flatten()
print('Sims range:', sims.min(), '-', sims.max())

# Top 5
top5 = np.argsort(sims)[::-1][:5]
print('Top 5:')
for idx in top5:
    pid = product_ids[idx]
    name = df_prod[df_prod['product_id'] == pid]['name'].values[0] if pid in df_prod['product_id'].values else 'N/A'
    print(f'  idx={idx}, pid={pid}, score={sims[idx]:.4f}, name={name[:40]}')

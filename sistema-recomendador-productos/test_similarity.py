import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from gensim.models import Word2Vec
import pandas as pd
from pathlib import Path

# Load
art = joblib.load(Path('models/hibrido_v2.pkl'))
w2v = Word2Vec.load(str(Path('models/word2vec_productos.model')))
data = np.load(Path('models/embeddings_word2vec.npz'), allow_pickle=True)
embeddings = data['embeddings']
df_prod = pd.read_csv(Path('data/productos.csv'))

print('Embeddings shape:', embeddings.shape)
print('Product IDs shape:', data['product_ids'].shape)

# Test similarity for a user
uid = 'u0007'
usuario_idx = art['idx_usuario'][uid]
print(f'Usuario idx: {usuario_idx}')

# Get some product indices from interactions
df_int = pd.read_csv(Path('data/interacciones.csv'))
hist = df_int[df_int['user_id'] == uid]['product_id'].values[:5]
print('Historial:', hist)

# Convert to indices
historial_idx = []
for pid in hist:
    if pid in art['idx_producto']:
        historial_idx.append(art['idx_producto'][pid])
print('Historial idx:', historial_idx)

# Calculate embedding average
emb_historial = embeddings[historial_idx].mean(axis=0)
print('Emb historial shape:', emb_historial.shape)

# Calculate similarity
sims = cosine_similarity(emb_historial.reshape(1, -1), embeddings).flatten()
print('Sims shape:', sims.shape)
print('Sims stats: min={:.4f}, max={:.4f}, mean={:.4f}'.format(sims.min(), sims.max(), sims.mean()))

# Top 5
top5 = np.argsort(sims)[::-1][:5]
print('Top 5 indices:', top5)
print('Top 5 scores:', sims[top5])
for idx in top5:
    name = df_prod.iloc[idx]['name']
    score = sims[idx]
    print(f'  {name[:40]} (score={score:.4f})')

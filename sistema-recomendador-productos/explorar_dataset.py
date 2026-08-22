import pandas as pd

df_train = pd.read_parquet('data/titulos-ecommerce-es/data/train-00000-of-00001.parquet')

print('=== CATEGORIAS ===')
print('Total categorias:', df_train['category'].nunique())
print()
print('Top 20 categorias:')
print(df_train['category'].value_counts().head(20))
print()
print('=== EJEMPLOS DE TITULOS ===')
for i in range(5):
    row = df_train.iloc[i]
    title = str(row['title'])[:80]
    cat = row['category']
    print(f'{i+1}. {title}...')
    print(f'   Categoria: {cat}')
    print()

# Check overlap with existing products
df_products = pd.read_csv('data/productos.csv')
print('=== DATOS EXISTENTES ===')
print('Productos existentes:', len(df_products))
print('Categorias existentes:', df_products['main_category'].nunique())
print()
print('Categorias existentes:')
print(df_products['main_category'].value_counts())

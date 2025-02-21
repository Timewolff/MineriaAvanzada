import pandas as pd

# Cargar el dataset
df = pd.read_csv('train-data.csv')

# Mostrar las primeras filas
print(df.head())

# Información general del dataset
print(df.info())

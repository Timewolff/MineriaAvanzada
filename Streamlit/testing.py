# MI TESTING

import model
import os
print(os.getcwd())

# 1. Cargar el archivo CSV con EDA
archivo_csv = os.path.join(os.path.dirname(__file__), "dataset", "diabetes_V2.csv")
eda = model.EDA(file=archivo_csv)

# 2. Pasar el EDA al optimizador
optimizador = model.DataOptimization(eda)

# 3. Ejecutar optimización y obtener los mejores parámetros
best_params = optimizador.opti_director(
    target_column='diabetes',         
    problem_type='classification',      
    method='both'                      
)

# 4. Crear la instancia del modelo supervisado pasando los mejores parámetros
modelo = model.Supervisado(optimizador, best_params=best_params)

# 5. Ejecutar los modelos de clasificación y comparar variantes
modelo.model_director(compare_params=True)

# 7. Visualizar la comparación usando ROC AUC
print("\nGenerando visualización comparativa...")
fig, ax = modelo.visualizar_comparacion_modelos(metrica='roc_auc', figsize=(10, 6))

# 8. Mostrar gráfico
import matplotlib.pyplot as plt
plt.show()
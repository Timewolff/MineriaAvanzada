import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda import EDA
from model import TimeSeriesModel

def show_supervised():
    st.title("Resultados del Modelo Supervisado")
    
    # Verificar si ya se han ejecutado los modelos
    if 'resultados' not in st.session_state:
        st.warning("No hay resultados disponibles. Por favor ejecuta el modelo primero desde la página 'EDA'.")
        return
    
    # Obtener los datos de la sesión
    resultados = st.session_state['resultados']
    modelo_supervisado = st.session_state['modelo_supervisado']
    problem_type = st.session_state['problem_type']
    
    # Generar y mostrar la visualización comparativa usando matplotlib
    st.header("Visualización Detallada de Modelos")
    if problem_type == 'classification':
        fig, ax = modelo_supervisado.visualizar_comparacion_modelos(metrica='roc_auc', figsize=(10, 6))
    else:
        fig, ax = modelo_supervisado.visualizar_comparacion_modelos(metrica='RMSE', figsize=(10, 6))
    
    st.pyplot(fig)
    
    # Mostrar una explicación de los resultados
    st.header("Interpretación de Resultados")
    
    if problem_type == 'classification':
        # Encontrar el mejor modelo basado en ROC AUC o accuracy
        if any(res.get('roc_auc') is not None for res in resultados):
            mejor_modelo = max([res for res in resultados if res.get('roc_auc') is not None], 
                              key=lambda x: x['roc_auc'])
            metrica_principal = 'ROC AUC'
            valor_metrica = mejor_modelo['roc_auc']
        else:
            mejor_modelo = max(resultados, key=lambda x: x['accuracy'])
            metrica_principal = 'accuracy'
            valor_metrica = mejor_modelo['accuracy']
    else:
        # Para regresión, el mejor modelo es el que tiene menor RMSE
        mejor_modelo = min(resultados, key=lambda x: x['RMSE'])
        metrica_principal = 'RMSE'
        valor_metrica = mejor_modelo['RMSE']
    
    st.subheader(f"Mejor modelo: {mejor_modelo['modelo']}")
    st.write(f"El mejor modelo según la métrica {metrica_principal} es **{mejor_modelo['modelo']}** con un valor de {valor_metrica:.4f}.")
    
    if '(Default)' in mejor_modelo['modelo']:
        st.write("Este modelo funciona mejor con sus parámetros predeterminados.")
    elif '(Genetic)' in mejor_modelo['modelo']:
        st.write("Este modelo fue optimizado usando algoritmos genéticos.")
    elif '(Exhaustive)' in mejor_modelo['modelo']:
        st.write("Este modelo fue optimizado usando búsqueda exhaustiva de parámetros.")
    
    # Mostrar todos los parámetros del modelo
    st.subheader("Parámetros optimizados")
    
    if 'best_params' in st.session_state:
        best_params = st.session_state['best_params']
        
        tabs = st.tabs(["Método Genético", "Método Exhaustivo"])
        
        with tabs[0]:
            if 'genetic' in best_params:
                for model_name, params in best_params['genetic'].items():
                    st.write(f"**{model_name}**:")
                    for param_name, value in params.items():
                        st.write(f"- {param_name}: {value}")
            else:
                st.write("No se utilizó el método genético.")
                
        with tabs[1]:
            if 'exhaustive' in best_params:
                for model_name, params in best_params['exhaustive'].items():
                    st.write(f"**{model_name}**:")
                    for param_name, value in params.items():
                        st.write(f"- {param_name}: {value}")
            else:
                st.write("No se utilizó el método exhaustivo.")
    
def show_forecast():
    st.title("Model Results")

    # Load the cleaned dataset from EDA
    eda = EDA()
    data = eda.get_df()

    if data is not None:
        st.subheader("📈 Time Series Forecasting")

        # Let user select columns for date and values
        date_col = st.selectbox("Select date column", data.columns)
        value_col = st.selectbox("Select value column", data.columns)

        try:
            # Prepare the time series from selected columns
            ts_data = data[[date_col, value_col]].dropna()
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
            ts_data = ts_data.sort_values(by=date_col)
            ts_data.set_index(date_col, inplace=True)
            series = ts_data[value_col]

            # Select model type and steps to forecast
            model_type = st.radio("Choose forecasting model", ["prophet", "xgboost"], horizontal=True)
            steps = st.slider("Forecast steps", 1, 30, 7)

            # Train and forecast
            model = TimeSeriesModel(ts_data.reset_index(), date_col, value_col, model_type)
            model.train()
            forecast = model.forecast(steps)

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 4))
            series.plot(ax=ax, label="Historical")
            forecast.plot(ax=ax, label="Forecast", linestyle="--", marker="o")
            ax.set_title(f"Forecast ({steps} steps) - Model: {model_type}")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Time series processing error: {e}")
    else:
        st.warning("No valid data found from EDA.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda import EDA
from model.time_series import TimeSeriesModel

def show_supervised():
    st.title("Resultados del Modelo Supervisado")
    
    if 'resultados' not in st.session_state:
        st.warning("No hay resultados disponibles. Por favor ejecuta el modelo primero desde la página 'EDA'.")
        return
    
    resultados = st.session_state['resultados']
    modelo_supervisado = st.session_state['modelo_supervisado']
    problem_type = st.session_state['problem_type']
    
    st.header("Visualización Detallada de Modelos")
    if problem_type == 'classification':
        fig, ax = modelo_supervisado.visualizar_comparacion_modelos(metrica='roc_auc', figsize=(10, 6))
    else:
        fig, ax = modelo_supervisado.visualizar_comparacion_modelos(metrica='RMSE', figsize=(10, 6))
    
    st.pyplot(fig)
    
    st.header("Interpretación de Resultados")
    
    if problem_type == 'classification':
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
    st.header("Forecast Results")

    df = st.session_state.get("forecast_df")
    date_col = st.session_state.get("date_col")
    value_col = st.session_state.get("value_col")

    if df is None or date_col is None or value_col is None:
        st.error("Missing forecast data.")
        return

    model_type = st.selectbox("Choose time series model", ["prophet", "xgboost"])
    steps = st.slider("Number of days to forecast", 1, 30, 7)

    if st.button("Run Forecast"):
        ts_model = TimeSeriesModel(df, date_col, value_col, model_type)
        ts_model.train()
        forecast = ts_model.forecast(steps)

        st.line_chart(forecast)
        st.success("Forecast completed.")

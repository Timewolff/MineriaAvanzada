import streamlit as st
from streamlit_vizzu import VizzuChart
import pandas as pd

def create_vizzu_chart(df, x_column, y_column, chart_type):
    """Genera una visualización interactiva con Vizzu en Streamlit."""
    
    # Asegurar que las columnas seleccionadas existen
    if x_column not in df.columns or y_column not in df.columns:
        st.error("Las columnas seleccionadas no existen en el DataFrame.")
        return

    # Filtrar el DataFrame solo con las columnas necesarias
    df_filtered = df[[x_column, y_column]].copy()

    # Convertir el DataFrame al formato adecuado
    df_dict = df_filtered.to_dict(orient="records")

    # Crear el gráfico correctamente
    chart = VizzuChart(df_filtered)

    # Configuración correcta para animate()
    animation = {
        "data": df_dict,  
        "config": {
            "channels": {
                "x": {"set": [x_column]},
                "y": {"set": [y_column]}
            },
            "geometry": chart_type  
        }
    }

    # Aplicar la animación correctamente
    chart.animate(**animation)  

    # Mostrar el gráfico en Streamlit
    st.write("### Gráfico Interactivo con Vizzu")
    st.write(chart)

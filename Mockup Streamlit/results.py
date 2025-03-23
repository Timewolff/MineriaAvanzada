import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


def show():
    # Generar datos sintéticos para el modelo
    np.random.seed(42)
    n_samples = 500
    año = np.random.randint(2000, 2023, size=n_samples)
    kilometraje = np.random.randint(5000, 200000, size=n_samples)
    precio = 20000 - (año - 2000) * 100 + np.random.normal(0, 3000, size=n_samples) - kilometraje * 0.05 + np.random.normal(0, 1000, size=n_samples)

    data = pd.DataFrame({'Año': año, 'Kilometraje': kilometraje, 'Precio': precio})

    # Crear un DataFrame para la matriz de correlación
    corr_matrix_df = data.corr()

    # Crear un DataFrame para los valores de R² de los modelos
    r2_values = [0.6887, 0.7561, 0.6888, 0.8690, 0.7030, 0.9121, 0.9121]
    model_name = [
        "Regresión Lineal Simple", "Support Vector Machine", "Regresión Ridge", 
        "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor", 
        "XGBoost Regressor"
    ]
    df_metrics = pd.DataFrame({"Model": model_name, "R²": r2_values})

    # Crear un DataFrame para los datos de Predicción vs Real
    num_points = 25
    x = np.linspace(1, num_points, num_points)
    real_data = np.sin(x / 3) + np.random.normal(0, 0.03, num_points)
    predict_data = real_data + np.random.normal(0, 0.11, num_points)
    chart_data_df = pd.DataFrame({
        "x": np.tile(x, 2),
        "y": np.concatenate([real_data, predict_data]),
        "Category": ["Real"] * num_points + ["Predicted"] * num_points
    })

    # Crear un DataFrame para los boxplots
    boxplot_df = data[['Precio']]

    # Crear un DataFrame para el gráfico de dispersión
    scatterplot_df = data[['Año', 'Precio']]

    # Crear un DataFrame para el gráfico de líneas adicional
    time_series_df = pd.DataFrame({
        "Año": np.arange(2000, 2023),
        "Ventas": np.random.randint(1000, 10000, size=23)
    })

    # Función para mostrar métricas
    def styled_metric(label, value):
        st.markdown(
            f"""
            <div style="text-align: center; line-height: 1.2;">
                <p style="margin-bottom: 8px; font-size: 24px; font-weight: bold;">{label}</p>
                <p style="font-size: 34px; color: #f4c542;">{value}</p>  <!-- Amarillo Simpsons -->
            </div>
            """,
            unsafe_allow_html=True
        )

    # Mostrar control deslizante al principio
    num_data_points = st.slider("Selecciona el número de puntos de datos a mostrar", min_value=10, max_value=100, value=50, key="slider", help="Desliza para seleccionar los puntos de datos que deseas visualizar", label_visibility="hidden")

    # Personalización del slider
    st.markdown("""
    <style>
        /* Cambiar color del slider */
        .streamlit-slider {
            background-color: #000000;  /* Color negro para el slider */
        }
        .stSlider .st-bb {
            background-color: #000000 !important;
        }
        .stSlider .st-b7 {
            background-color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Mostrar métricas
    row1_col1, row1_col2 = st.columns([1.2, 1.2])
    with row1_col1:
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        with st.container():
            subcol1, subcol2 = st.columns([1, 1])
            with subcol1:
                styled_metric("Modelo", "Regresión Lineal")
            with subcol2:
                styled_metric("Mejor modelo", "XGBoost Regressor")

        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                styled_metric("R²", "0.912")
            with col2:
                styled_metric("RMSE", "3.619")
            with col3:
                styled_metric("MAE", "1.967")

    # Gráfico 1: Métricas de Modelos
    row1_col2, row2_col1 = st.columns([1, 1])
    with row1_col2:
        st.header("Metric: R²")
        chart = alt.Chart(df_metrics.head(num_data_points)).mark_bar().encode(
            x='Model',
            y='R²',
            tooltip=['Model', 'R²'],
            color=alt.value("#f4c542")  # Amarillo Simpsons
        ).properties(
            width=500,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

    # Gráfico 2: Predicción vs Datos Reales
    with row2_col1:
        st.header("Predicted Model vs Real Data")
        color_scale = alt.Scale(domain=["Real", "Predicted"], range=["#00c2b8", "#f4c542"])  # Celeste y Amarillo
        chart = alt.Chart(chart_data_df.head(num_data_points)).mark_line().encode(
            x=alt.X("x"),
            y=alt.Y("y"),
            color=alt.Color("Category", scale=color_scale, legend=alt.Legend(title="Legend")),
            tooltip=['x', 'y', 'Category']
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

    # Gráfico 3: Matriz de Correlación
    row1_col2, row2_col2 = st.columns(2)
    with row1_col2:
        st.header("Matriz de Correlación")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, cbar_kws={'label': 'Correlación'})
        st.pyplot(fig)

    # Gráfico 4: Boxplot de Precio
    with row2_col2:
        st.header("Distribución del Precio (Boxplot)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=boxplot_df.head(num_data_points)['Precio'], color='#00c2b8', ax=ax2)  # Celeste
        st.pyplot(fig2)

    # Gráfico 5: Gráfico de Dispersión entre Año y Precio
    with row2_col2:
        st.header("Relación entre Año y Precio")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=scatterplot_df.head(num_data_points)['Año'], y=scatterplot_df.head(num_data_points)['Precio'], color='#f4c542', ax=ax3)  # Amarillo
        st.pyplot(fig3)

    # Gráfico 6: Gráfico de Líneas de Ventas a lo largo de los años (Nuevo)
    row1_col1, row2_col1 = st.columns([1, 1])
    with row1_col1:
        st.header("Evolución de Ventas a lo Largo del Tiempo")
        chart = alt.Chart(time_series_df.head(num_data_points)).mark_line().encode(
            x=alt.X('Año', title='Año'),
            y=alt.Y('Ventas', title='Ventas'),
            tooltip=['Año', 'Ventas'],
            color=alt.value("#00c2b8")  # Celeste
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

    # Gráfico 7: Gráficos Interactivos con Vizzu
    st.subheader("Gráficos Interactivos con Vizzu")

    # Selección de columnas para los ejes
    x_column = st.selectbox("Selecciona el eje X", data.columns)
    y_column = st.selectbox("Selecciona el eje Y", data.columns)

    # Selección del tipo de gráfico
    chart_type = st.selectbox("Selecciona el tipo de gráfico", ["bar", "line"])

show()

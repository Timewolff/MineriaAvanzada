import streamlit as st
import streamlit_antd_components as sac
from streamlit_option_menu import option_menu

# Import other modules
import about
import eda
import results

main_color = "#384B70"

# Main App
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Home", "EDA", "Results", "About"],  # Se agrega Home
        icons=["house", "bar-chart", "rocket-takeoff", "bi bi-mortarboard"],  # Icono para Home
        menu_icon="cast",
        default_index=0,  # Home será la primera opción
        styles={
            "container": {"background-color": "#f8f9fa"},
            "icon": {"font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#d9d9d9"},
            "nav-link-selected": {"background-color": main_color, "color": "white", "icon": {"color": "white"} },
        }
    )

# -------------------------
# ESTA ES UNA CORRECCION DEL PROFE -Seccion Home
# -------------------------
if selected == "Home":
    st.title("Bienvenido a Mockup Paquete de Python")

    # Versión del Proyecto
    st.subheader(" Versión del Proyecto")
    st.markdown("""
    - **Versión 1.0**
    - Última actualización: Febrero 2025
    """)
    
      # Mejoras Futuras
    st.subheader("Mejoras Futuras")
    st.markdown("""
    ✅ Optimización de modelos de predicción.  
    ✅ Mejora en la interfaz de usuario para mayor interactividad.  
    ✅ Agregar más tipos de gráficos para análisis exploratorio.  
    ✅ Implementación de animaciones con Vizzu *(en progreso)*.  
    """)

elif selected == "EDA":
    from eda import EDA

    st.title("Exploración de Datos - EDA")

    # Crear una instancia de la clase EDA con el dataset
    eda_instance = EDA('test-data.csv')

    # Tabla de contenido
    st.subheader("Contenido")
    st.write("1. Primeras Filas del Dataset")
    st.write("2. Tipos de Datos")
    st.write("3. Valores Atípicos")
    st.write("4. Gráficos")

    # Primeras Filas del Dataset
    st.subheader("Primeras Filas del Dataset")
    st.write(eda_instance.head_df())

    # Tipos de Datos
    st.subheader("Tipos de Datos")
    st.write(eda_instance.check_data_types())

    # Valores Atípicos
    st.subheader("Valores Atípicos")
    st.write(eda_instance.detect_outliers())

    # Gráficos en Streamlit
    st.subheader("Gráficos")
    with st.expander("Histogramas"):
        columna = st.selectbox("Selecciona una columna para visualizar el histograma", eda_instance.get_df().columns)
        eda_instance.plot_histogram(columna)

    with st.expander("Gráficos de Dispersión"):
        col1 = st.selectbox("Selecciona la primera variable", eda_instance.get_df().columns)
        col2 = st.selectbox("Selecciona la segunda variable", eda_instance.get_df().columns)
        eda_instance.plot_scatter(col1, col2)

    with st.expander("Mapa de Calor"):
        eda_instance.plot_heatmap()

    with st.expander("Gráfico de Barras"):
        col = st.selectbox("Selecciona una columna para el gráfico de barras", eda_instance.get_df().columns)
        eda_instance.plot_bar(col)

    with st.expander("Diagrama de Violin"):
        col_violin = st.selectbox("Selecciona una columna para el diagrama de violín", eda_instance.get_df().columns)
        eda_instance.plot_violin(col_violin)


    with st.expander("Gráfico Lineal"):
        eda_instance.plot_line()

    with st.expander("Gráfico por Pares"):
        eda_instance.plot_pairplot()

    with st.expander("Diagrama de Caja"):
        eda_instance.plot_boxplot()

elif selected == "Results":
    results.show()
    
elif selected == "About":
    about.show()



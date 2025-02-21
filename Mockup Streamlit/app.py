import streamlit as st
import streamlit_antd_components as sac
from streamlit_option_menu import option_menu

# Import other modules
import home
import eda
import results

# Main menu
selected = option_menu(
    menu_title=None,
    options=["Home", "EDA", "Results"],
    icons=["house", "bar-chart", "rocket-takeoff"],  # Bootstrap Icons
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={  # 🎨 Personalización de estilos
        "container": {
            "padding": "0!important",
            "background-color": "#292929",
        },
        "icon": {"font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px 10px",
            "color": "#fff",
            "border-radius": "10px",
        },
        "nav-link-selected": {
        "background-color": "#ffa31a",
        "color": "#000000",
        "font-weight": "bold",
        "icon": {"color": "#000000"},
    },
    }
)

if selected == "Home":
    home.show()
elif selected == "EDA":
    from eda import EDA  # Importamos la clase EDA

    st.title("Exploración de Datos - EDA")

    # Crear una instancia de la clase EDA con el dataset
    eda_instance = EDA('test-data.csv')

    st.subheader("Primeras Filas del Dataset")
    st.write(eda_instance.head_df())

    st.subheader("Tipos de Datos")
    st.write(eda_instance.check_data_types())

    st.subheader("Valores Atípicos")
    st.write(eda_instance.detect_outliers())

    # Gráficos en Streamlit
    st.subheader("Histogramas")
    columna = st.selectbox("Selecciona una columna para visualizar el histograma", eda_instance.get_df().columns)
    eda_instance.plot_histogram(columna)

    st.subheader("Gráficos de Dispersión")
    col1 = st.selectbox("Selecciona la primera variable", eda_instance.get_df().columns)
    col2 = st.selectbox("Selecciona la segunda variable", eda_instance.get_df().columns)
    eda_instance.plot_scatter(col1, col2)

    st.subheader("Mapa de Calor")
    eda_instance.plot_heatmap()

elif selected == "Results":
    results.show()
import streamlit as st

def display_section(header, content):
    st.header(header)
    st.markdown(content)

def show():
    st.markdown('<h1 style="color:#384B70;">Mockup Paquete de Python</h1>', unsafe_allow_html=True)
    
    # Descripción del proyecto
    st.header("Descripción del Proyecto:")
    st.markdown("""
    Proyecto que utiliza regresión y *machine learning* para predecir el precio de autos usados según sus características.
    """)
    
    # Descripción del dataset
    display_section(
        "Dataset: Used Cars Price Prediction",
        """
        Dataset para predecir el precio de autos usados, basado en características como marca, año y kilometraje. 
        Obtén más detalles en Kaggle: [Used Cars Price Prediction](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction?select=train-data.csv)
        """
    )
    
    # Integrantes del grupo
    display_section(
        'Integrantes del grupo:',
        """
        - Carolina Salas Moreno 
        - Deykel Bernard Salazar
        - Esteban Ramirez Montano
        - Kristhel Porras Mata
        - Marla Gomez Hernández
        """
    )
    st.write("¡El mejor equipo! 💪🔥")

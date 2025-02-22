import streamlit as st
import pandas as pd

def display_section(header, content):
    st.header(header)
    st.markdown(content)

def show():
    st.title('Mockup Paquete de Python')

    col1, col2 = st.columns([1.2, 1])

    # Integrantes del grupo
    with col1:
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

    with col2:
        # Imagen de Homero
        st.image("Recursos/homero_pensando.jpg", width=200)

    # Descripción del proyecto
    st.header("🚗 Descripción del Proyecto:")
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

    # Crear una tabla con las columnas del dataset
    columns_data = {
        "Columna": ["Name", "Location", "Year", "Kilometers_Driven", "Fuel_Type", "Transmission", "Owner_Type", "Mileage", "Engine", "Power", "Torque", "Seats", "Price"],
        "Descripción": [
            "Marca y modelo del auto.",
            "Ubicación de venta del auto.",
            "Año del modelo.",
            "Kilómetros recorridos.",
            "Tipo de combustible (Petrol/Diesel/etc.).",
            "Tipo de transmisión (Auto/Manual).",
            "Tipo de propietario.",
            "Kilometraje estándar (km/l).",
            "Volumen del motor (cc).",
            "Potencia del motor (bhp).",
            "Torque máximo (nm).",
            "Número de asientos.",
            "Precio del auto (a predecir)."
        ]
    }

    # Convertir los datos a un DataFrame para mejor presentación
    df_columns = pd.DataFrame(columns_data)

    # Estilizar la tabla
    styled_table = df_columns.style.set_properties(**{'text-align': 'left', 'padding': '5px'}) \
                                   .set_table_styles([{'selector': 'th', 'props': [('background-color', '#F8D000'), ('color', 'white'), ('font-weight', 'bold')]}])

    # Mostrar la tabla con el estilo
    st.table(styled_table)

if __name__ == "__main__":
    show()

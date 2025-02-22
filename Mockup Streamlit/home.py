import streamlit as st
def show():
    st.title('Mockup paquete de Python ')
    #Adjust columns width
    col1, col2 = st.columns([1.2, 1])  # 50% - 50%

    with col1:
        st.header('👥 Integrantes del grupo:')
        st.markdown("""
        - Esteban Ramirez Montano
        - Kristhel Porras Mata
        - Deykel Bernard Salazar
        - Marla Gomez H
        - Carolina Salas
        """)
        st.write("¡El mejor equipo! 💪🔥")

    with col2:
        st.image("Recursos/homero_pensando.jpg")

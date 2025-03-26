import streamlit as st

def display_section(header, content):
    st.header(header)
    st.markdown(content)

def show():
    st.markdown('<h1 style="color:#384B70;">Python ModelBoard Project</h1>', unsafe_allow_html=True)
    st.markdown("""
    Using Machine Learning models, ModelBoard allows you to evaluate various prediction and classification algorithms and choose the best model for your project.
    """)
    st.markdown("""
    - **ModelBoard Version 1.1**
    - Last update: March 2025
    """)
    
    st.subheader("Future Improvements")
    st.markdown("""
    ✅ Optimization of predictive models.  
    ✅ Improved user interface for greater interactivity.  
    ✅ Add more types of charts for exploratory analysis.  
    ✅ Implementation of animations with Vizzu *(in progress)*.  
    """)
    
    # Dataset description
    display_section(
        "Dataset: Used Cars Price Prediction",
        """
        Dataset for predicting the price of used cars, based on features such as brand, year, and mileage. 
        Get more details on Kaggle: [Used Cars Price Prediction](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction?select=train-data.csv)
        """
    )
    
    # Group members
    display_section(
        'Group Members:',
        """
        - Carolina Salas Moreno 
        - Deykel Bernard Salazar
        - Esteban Ramirez Montano
        - Kristhel Porras Mata
        - Marla Gomez Hernández
        """
    )
    st.write("The best team! 💪🔥")

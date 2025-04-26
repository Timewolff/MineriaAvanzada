import streamlit as st

def display_section(header, content):
    st.header(header)
    st.markdown(content)

def show():
    st.markdown('<h1 style="color:#384B70;">About</h1>', unsafe_allow_html=True)
    st.markdown("""
    This project is an interactive web application developed with Streamlit that facilitates data analysis and implementation of machine learning models. Designed to simplify the data science process from exploration to prediction, and is accessible to both novice and experienced analysts.
    """)
    st.markdown("""
    ### Project Features:
    - **Data Exploration**: Visualize and analyze datasets with various charts and graphs.
    - **Supervised Models**: Automated implementation of classification and regression algorithms with hyperparameter optimization using genetic and exhaustive search.
    - **Forecasting**: Forecasting capabilities using Prophet and ARIMA, ideal for predicting future trends in sequential data.
    - **Deep Learning**: Coming soon!
    """)
    
    st.subheader("Evaluation Metrics Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Classification
        - **Accuracy:** Proportion of all classifications that were correct. Higher values indicate better performance.
        - **Precision:** Proportion of positive predictions that were correct. Higher values indicate better performance.
        - **Recall:** Proportion of actual positive instances that were correctly identified. Higher values indicate better performance.
        - **F1 Score:** Harmonic mean of precision and completeness, combining precision and recall, to obtain a much more objective value. Higher values indicate better performance.
        - **ROC AUC:** Area under the curve of True Positive Rate versus False Positive Rate at various thresholds. Higher values indicate better performance.
        """)
    with col2:
        st.markdown("""
        ### Regression
        - **Mean Absolute Error (MAE):** Average of the absolute differences between predicted and actual values.
        - **Mean Squared Error (MSE):** Average of squared differences between predicted and actual values. Better predictions get a result close to *0*.
        - **Root Mean Squared Error (RMSE):** Square root of the average of the squared differences between predicted and actual values. Better predictions get a result close to *0*.
        - **R-squared (R²):** Proportion of the variance in the dependent variable that is predictable from the independent variables.Values close to 1, the model explains well the proportion of the variance. (overfitting in some cases).
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

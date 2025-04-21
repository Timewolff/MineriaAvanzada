import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from model import EDA

def show():
    """
    Show a quick exploratory data analysis (EDA) of the dataset analyzed.

    Required:
    st.session_state['eda'] obtained from start.py
    
    Actions:
    1. Checks if the EDA session state exists.
    2. Creates a tab menu where a part of the dataset
    3. Detects outliers and displays them in a table.
    4. Displays a histogram of a selected column.
    5. Displays a scatter plot of two selected variables.
    6. Displays a heatmap of the dataset.
    """
    # 1. Checks if the EDA session state exists.
    if 'eda' not in st.session_state:
        st.warning("No data available. You need to run the model first in Start Analysis option.")
        return
    # Use the session state
    eda_instance = st.session_state['eda']
    
    # Just for formatting the tab component
    st.markdown("""
        <style>
            /* Style all tabs */
            div[data-testid="stTabs"] button {
                color: #384B70 !important; /* Blue text */
                font-weight: bold !important; /* Bold text */
            }

            /* Ensure font size applies correctly */
            div[data-testid="stTabs"] button p {
                font-size: 18px !important; /* Larger text */
                margin: 0px !important; /* Remove unnecessary margins */
            }

            /* Active tab underline color */
            div[data-testid="stTabs"] button[aria-selected="true"] {
                border-bottom: 3px solid #384B70 !important; /* Blue underline */
            }
        </style>
        """, unsafe_allow_html=True)

    st.title("Exploratory data analysis (EDA)")

    st.subheader("Extract from the dataset")
    # 2. Creates a tab menu where a part of the dataset
    tab1, tab2, tab3 = st.tabs(["Top Rows", "Last Rows", "Data Types"])
    with tab1:
        st.subheader("First 5 rows of the Dataset")
        st.write(eda_instance.head_df())
    with tab2:
        st.subheader("Last 5 rows of the Dataset")
        st.write(eda_instance.tail_df())
    with tab3:
        st.subheader("Data types")
        st.write(eda_instance.check_data_types())

    col1, col2 = st.columns(2)

    # 3. Detects outliers and displays them in a table.
    with col1:
        st.subheader("Detected Outliers")
        outliers_dict = eda_instance.detect_outliers()
        if isinstance(outliers_dict, dict):
            df_outliers = pd.DataFrame(outliers_dict.items(), columns=["Feature", "Outlier Count"])
            st.table(df_outliers)
        else:
            st.write(outliers_dict)

    # 4. Displays a histogram of a selected column.        
    with col2:
        columna = st.selectbox("Select a column to display the histogram", eda_instance.get_df().columns)
        eda_instance.plot_histogram(columna)

    col1, col2 = st.columns(2)

    # 5. Displays a scatter plot of two selected variables.
    with col1:
        st.subheader("Scatter Plot")
        selected_vars = st.multiselect(
            "Select two variables",
            eda_instance.get_df().columns,
            default=eda_instance.get_df().columns[:2].tolist() if len(eda_instance.get_df().columns) >= 2 else [],
            max_selections=2
        )
        if len(selected_vars) == 2:
            eda_instance.plot_scatter(selected_vars[0], selected_vars[1])
        else:
            st.warning("Please select exactly 2 variables.")

    # 6. Displays a heatmap of the dataset.
    with col2:
        st.subheader("Heatmap")
        eda_instance.plot_heatmap()
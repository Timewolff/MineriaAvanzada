import streamlit as st
import pandas as pd

def show():
    # Main columns
    row1_col1, row1_col2 = st.columns([1.5, 1.5])
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Algorithm family: Supervised")
        st.subheader("Best model: XGBoost Regressor")
        
        # Subcolumns
        col1, col2, col3 = st.columns([0.3, 0.1, 0.3])

        # Funtion to style metrics
        def styled_metric(label, value):
            st.markdown(
                f"""
                <div style="text-align: center; line-height: 1;">
                    <p style="margin-bottom: 1px; font-size: 14px; font-weight: bold;">{label}</p>
                    <p style="font-size: 26px; color: #ffa31a;">{value}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col1:
            styled_metric("R²", "0.912")

        with col2:
            styled_metric("RMSE", "3.619")

        with col3:
            styled_metric("MAE", "1.967")

    with row1_col2:
        st.header("Metric: R2")

        r2_values = [0.6887, 0.7561, 0.6888, 0.8690, 0.7030, 0.9121, 0.9121]
        model_name = ["Regresión Lineal Simple", "Support Vector Machine", "Regresión Ridge", 
        "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor", 
        "XGBoost Regressor"]
        df_metrics = pd.DataFrame({"Model": model_name, "R²": r2_values})

        st.bar_chart(df_metrics, x="Model", y="R²", color="#ffaa00", stack=False)

    with row2_col1:
        st.header("Título 3")

    with row2_col2:
        st.header("Título 4")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def show():
    # Main columns
    row1_col1, row1_col2 = st.columns([1.2, 1.2])
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        def styled_metric(label, value):
            st.markdown(
                f"""
                <div style="text-align: center; line-height: 1.2;">
                    <p style="margin-bottom: 8px; font-size: 24px; font-weight: bold;">{label}</p>
                    <p style="font-size: 34px; color: #ffa31a;">{value}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Model Basic Information
        st.markdown("<br>" * 3, unsafe_allow_html=True)

        # Container of vertical metrics
        with st.container():
            subcol1, subcol2 = st.columns([1, 1])

            with subcol1:
                styled_metric("Family", "Supervised")

            with subcol2:
                styled_metric("Best model", "XGBoost Regressor")

        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                styled_metric("R²", "0.912")

            with col2:
                styled_metric("RMSE", "3.619")

            with col3:
                styled_metric("MAE", "1.967")

    with row1_col2:
        st.header("Metric: R²")

        r2_values = [0.6887, 0.7561, 0.6888, 0.8690, 0.7030, 0.9121, 0.9121]
        model_name = [
            "Regresión Lineal Simple", "Support Vector Machine", "Regresión Ridge", 
            "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor", 
            "XGBoost Regressor"
        ]

        df_metrics = pd.DataFrame({"Model": model_name, "R²": r2_values})

        st.bar_chart(df_metrics, x="Model", y="R²", color="#ffa31a", stack=False)

    with row2_col1:
        st.header("Predicted Model vs Real Data")

        #Just for show the line chart
        num_points = 25
        x = np.linspace(1, num_points, num_points)
        real_data = np.sin(x / 3) + np.random.normal(0, 0.03, num_points)
        predict_data = real_data + np.random.normal(0, 0.11, num_points)

        # Its required to create a DataFrame to plot the line chart
        chart_data = pd.DataFrame({
            "x": np.tile(x, 2),
            "y": np.concatenate([real_data, predict_data]),
            "Category": ["Real"] * num_points + ["Predicted"] * num_points
        })

        # Some customizations
        color_scale = alt.Scale(domain=["Real", "Predicted"], range=["white", "#ffa31a"])

        #line chart
        chart = alt.Chart(chart_data).mark_line().encode(
            x=alt.X("x"),
            y=alt.Y("y"),
            color=alt.Color("Category", scale=color_scale, legend=alt.Legend(title="Legend"))
        ).properties(
            width=600,
            height=400
        )

        # Mostrar gráfico en Streamlit
        st.altair_chart(chart, use_container_width=True)


    with row2_col2:
        st.header("Título 4")

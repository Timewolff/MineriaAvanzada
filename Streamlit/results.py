import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from eda import EDA
#Por arreglar
#from model.time_series import TimeSeriesModel

def show_supervised():
    # Load session data
    resultados = st.session_state['resultados']
    problem_type = st.session_state['problem_type']

    # Select metric based on problem type
    metric = 'roc_auc' if problem_type == 'classification' else 'RMSE'

    # Prepare values by model and optimization type
    data_dict = {}
    for resultado in resultados:
        nombre_completo = resultado.get('modelo')
        if metric not in resultado:
            continue
        if '(' in nombre_completo:
            nombre_base, optimizacion = nombre_completo.split('(')
            optimizacion = optimizacion.replace(')', '').strip()
        else:
            nombre_base = nombre_completo
            optimizacion = 'Default'
        nombre_base = nombre_base.strip()
        data_dict.setdefault(nombre_base, {})[optimizacion] = resultado[metric]

    # Build DataFrame
    modelos = sorted(data_dict.keys())
    df_chart = pd.DataFrame({opt: [data_dict.get(m, {}).get(opt, 0) for m in modelos]
                             for opt in ['Default', 'Genetic', 'Exhaustive']},
                            index=modelos)

    if metric == 'RMSE':
        st.caption("Note: For RMSE, lower values are better.")
    else:
        chart_metric = "AUC"

    # Reshape data for Altair
    df_long = df_chart.reset_index().melt(id_vars='index', var_name='Optimization', value_name='Value')
    df_long.rename(columns={'index': 'Model'}, inplace=True)

    st.subheader("Model Performance Comparison Between Optimization Methods")
    # Define custom colors
    color_scale = alt.Scale(domain=['Default', 'Genetic', 'Exhaustive'],
                            range=['#384B70', '#B8001F', '#507687'])

    # Main bar chart
    bars = alt.Chart(df_long).mark_bar().encode(
        x=alt.X('Model:N', title='Model',
            axis=alt.Axis(
                labelAngle=0,
                labelFontSize=16,
                titleFontSize=16,
                labelLimit=0
            )
        ),
        y=alt.Y('Value:Q',
            axis=alt.Axis(labels=False, ticks=False, title=chart_metric, titleFontSize=18)),
        color=alt.Color('Optimization:N', scale=color_scale,
                        legend=alt.Legend(title="Parameters")),
        xOffset='Optimization:N',
        tooltip=['Model', 'Optimization', 'Value']
    )

    # Text on top of bars
    text = alt.Chart(df_long).mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=13
    ).encode(
        x=alt.X('Model:N'),
        xOffset=alt.X('Optimization:N'),
        y=alt.Y('Value:Q'),
        text=alt.Text('Value:Q', format=".3f"),
        tooltip=['Model', 'Optimization', 'Value']
    )

    # Combine and render
    chart = (bars + text).properties(
        width=900,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)



                
def show_forecast():
    st.header("Forecast Results")

    df = st.session_state.get("forecast_df")
    date_col = st.session_state.get("date_col")
    value_col = st.session_state.get("value_col")

    if df is None or date_col is None or value_col is None:
        st.error("Missing forecast data.")
        return

    model_type = st.selectbox("Choose time series model", ["prophet", "xgboost"])
    steps = st.slider("Number of days to forecast", 1, 30, 7)

    if st.button("Run Forecast"):
        pass
        '''ts_model = TimeSeriesModel(df, date_col, value_col, model_type)
        ts_model.train()
        forecast = ts_model.forecast(steps)

        st.line_chart(forecast)
        st.success("Forecast completed.")'''

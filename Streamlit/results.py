import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import model
import os
import json
from eda import EDA
from sklearn.metrics import confusion_matrix
#Needs revision
#from model.time_series import TimeSeriesModel

def build_dataframe(resultados, problem_type):
    metric = 'roc_auc' if problem_type == 'classification' else 'RMSE'
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

    modelos = sorted(data_dict.keys())
    df_chart = pd.DataFrame({opt: [data_dict.get(m, {}).get(opt, 0) for m in modelos]
                             for opt in ['Default', 'Genetic', 'Exhaustive']},
                            index=modelos)
    return df_chart, metric

def show_supervised():
    # Load session data
    resultados = st.session_state['resultados']
    problem_type = st.session_state['problem_type']

    df_chart, metric = build_dataframe(resultados, problem_type)
    chart_metric = "RMSE" if metric == 'RMSE' else "AUC"

    # Reshape data for Altair
    df_long = df_chart.reset_index().melt(id_vars='index', var_name='Optimization', value_name='Value')
    df_long.rename(columns={'index': 'Model'}, inplace=True)

    st.subheader("Performance Breakdown by Optimization Strategy")
    # Custom colors
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

    # Data for grid of best metrics
    if metric == 'roc_auc':
        best_model_tuple = df_chart.stack().idxmax()
        best_model_name = best_model_tuple[0]
        best_auc_value = round(df_chart.loc[best_model_tuple[0], best_model_tuple[1]], 3)
        best_optimization = best_model_tuple[1]

    #New variable to find other metrics
    good_name_model = f"{best_model_name} ({best_optimization})"

    # Find the best model metrics to show in the grid
    best_finder_metrics = next((r for r in resultados if r.get("modelo") == good_name_model), None)
    if best_finder_metrics:
        accuracy = round(best_finder_metrics.get("accuracy", 0), 3)
        precision = round(best_finder_metrics.get("precision", 0), 3)
        f1_score = round(best_finder_metrics.get("f1_score", 0), 3)

    # First column level
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Summary: {best_model_name}")
        a, b = st.columns(2)
        c, d = st.columns(2)

        a.metric("AUC", best_auc_value, border=True)
        b.metric("Accuracy", accuracy, border=True)
        c.metric("Precision", precision, border=True)
        d.metric("F1 Score", f1_score, border=True)
        st.caption("*The best algorithm was selected using AUC")
    with col2:
        st.subheader("Peformance Metrics by Model")

        modelo = st.session_state['modelo_supervisado']
        df_exhaustivo = modelo.get_exhaustive_metrics()

        if not df_exhaustivo.empty:
            styled_df = df_exhaustivo.style.apply(
                lambda s: ['background-color: #c2c7d8' if v == s.max() else '' for v in s],
                axis=0,
                subset=df_exhaustivo.columns[1:]  # omitir 'Modelo'
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.warning("No exhaustive results available.")
        st.caption("Results obtained using only exhaustive search.")
    
    # Predictions vs Real and processing time

    # Dataframe for predictions vs real
    best_model = next((r for r in resultados if r['modelo'] == good_name_model), None)

    # Second column level
    col1, col2 = st.columns(2)

    with col1:
        if best_model and 'real_values' in best_model and 'predicted_values' in best_model:
            real = best_model['real_values']
            pred = best_model['predicted_values']

            with col1:
                st.subheader("Confusion Matrix")
                labels = sorted(list(set(real + pred)))
                cm = confusion_matrix(real, pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=labels,yticklabels=labels, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                st.caption("Evaluation based on the model with the highest AUC under exhaustive search.")
        else:
            st.warning("Confusion matrix data is not available for this model.\n Evaluation based on the model with the highest AUC under exhaustive search.")

    with col2:
        st.subheader("Optimization Execution Time")
        # Load execution times from several JSON files inside "dataset" folder
        file_path = os.path.join('dataset', 'DO_execution_time.json')
        with open(file_path, 'r') as f:
                optimization_execution_times = json.load(f)

        file_path = os.path.join('dataset', 'DM_execution_time.json')
        with open(file_path, 'r') as f:
                model_execution_times = json.load(f)
        
        # Select box for choosing chart about execution time
        select_box_option = st.selectbox(
            "Select a chart to view:",
            ("Data Optimization Time", "Algorithm Execution Time"),
            index=None,
            placeholder="Choose an option...",
        )

        if select_box_option == "Data Optimization Time":
            optimization_times = optimization_execution_times.copy()
            total_optimization_time = optimization_times.pop('Optimization', None)
            
            # Convert dictionary to DataFrame for Altair
            times_df = pd.DataFrame({
                'Optimization': list(optimization_times.keys()),
                'Time': list(optimization_times.values())
            })
            
            # Capitalize method names for better presentation
            times_df['Optimization'] = times_df['Optimization'].str.capitalize()
            
            # Create vertical bar chart with improved styling
            bars = alt.Chart(times_df).mark_bar().encode(
                x=alt.X('Optimization:N', 
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=16,
                        title=None
                    )
                ),
                y=alt.Y('Time:Q',
                    axis=alt.Axis(title='Time (seconds)', titleFontSize=18)),
                color=alt.Color('Optimization:N', 
                            scale=alt.Scale(domain=['Genetic', 'Exhaustive'],
                                            range=['#B8001F', '#507687']),
                            legend=None),
                tooltip=['Optimization', 'Time']
            )
            
            # Text on top of bars
            text = alt.Chart(times_df).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontSize=13,
                fontWeight='bold'
            ).encode(
                x=alt.X('Optimization:N'),
                y=alt.Y('Time:Q'),
                text=alt.Text('Time:Q', format=".2f"),
                tooltip=['Optimization', 'Time']
            )
            
            # Combine and render
            chart = (bars + text).properties(
                height=400
            )
            
            # Display chart in Streamlit
            st.altair_chart(chart, use_container_width=True)
            
            # Display total optimization time as a metric
            if total_optimization_time:
                st.metric("Total Optimization Processing Time", f"{total_optimization_time:.2f} seconds")
            st.caption("Execution time for each optimization strategy in seconds")

        elif select_box_option == "Algorithm Execution Time":
            algorithm_times = model_execution_times.copy()
            
            # Remove 'Total' from the dictionary
            total_time = algorithm_times.pop('Total', None)
            
            # Convert dictionary to DataFrame for Altair Graph
            algo_times_df = pd.DataFrame({
                'Algorithm': list(algorithm_times.keys()),
                'Time': list(algorithm_times.values())
            })
            
            # Sort by execution time in descending order for better visualization
            algo_times_df = algo_times_df.sort_values('Time', ascending=False)
            
            # Calculate the number of unique algorithms
            unique_algorithms = len(algo_times_df)
            
            # Color palette for the bars
            color_set = ['#384B70', '#B8001F', '#3F51B5', '#C9B27C', '#768750', '#001FB8', '#00BCD4', '#4CAF50']
            
            # Create vertical bar chart
            bars = alt.Chart(algo_times_df).mark_bar().encode(
                x=alt.X('Algorithm:N', 
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=14,
                        title=None
                    )
                ),
                y=alt.Y('Time:Q',
                    axis=alt.Axis(title='Time (seconds)', titleFontSize=18)),
                color=alt.Color('Algorithm:N', 
                            scale=alt.Scale(range=color_set[:unique_algorithms]),
                            legend=None),
                tooltip=['Algorithm', 'Time']
            )
            
            # Text on top of bars
            text = alt.Chart(algo_times_df).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontSize=13,
                fontWeight='bold'
            ).encode(
                x=alt.X('Algorithm:N'),
                y=alt.Y('Time:Q'),
                text=alt.Text('Time:Q', format=".2f"),
                tooltip=['Algorithm', 'Time']
            )
            
            # Combine and render
            chart = (bars + text).properties(
                height=400
            )
            
            # Display chart in Streamlit
            st.altair_chart(chart, use_container_width=True)
            # Display total time
            if total_time:
                st.metric("Supervised Learning Processing Time", f"{total_time:.2f} seconds")
            st.caption("Execution time for each algorithm in seconds")
        else:
            st.warning("Please select a valid option from the dropdown.")

from prophet import Prophet

def show_forecast():
    st.header("Forecast Results")

    df = st.session_state.get("forecast_df")

    # Confirm and assign correct datetime and value columns
    date_col = "fecha"  #This is the column created with FakeTimeGenerator
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    value_col = st.selectbox("Select the value to forecast", numeric_cols)

    if df is None or date_col is None or value_col is None:
        st.error("Missing forecast data.")
        return
    # Create a new column for the forecasted values
    steps = st.slider("Number of days to forecast", 1, 30, 7)

    if st.button("Run Forecast"):
        try:
            df_prophet = df[[date_col, value_col]].copy()
            df_prophet.columns = ['ds', 'y']

            df_prophet['ds'] = pd.to_datetime(df_prophet['ds']) 

            # Fit and predict
            m = Prophet()
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=steps)
            forecast = m.predict(future)

            # Plot
            fig = m.plot(forecast)
            st.pyplot(fig)
            st.success("Forecast completed.")
            st.write("Forecast Data", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps))
        except Exception as e:
            st.error(f"Forecasting failed: {e}")

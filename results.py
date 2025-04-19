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
from prophet import Prophet

def show_supervised():
    """
    Display results obtained from the supervised learning model

    Required:
    st.session_state['resultados'] and ['problem_type'] obtained from start.py
    
    Actions:
    1. Creates df_metrics DataFrame from the results stored in session state
    2. Shows a bar chart of AUC (classification)or RMSE (regression) by
       optimization strategy
    3. Shows a summary of the best model's key metrics using only exhaustive
       optimization
    4. Shows a confusion matrix or comparison of predicted vs actual values
       depending on the problem type
    5. Shows time exacution on the optimization and algorithm execution
    """

    # Load stored results and problem type
    results = st.session_state['resultados']
    problem_type = st.session_state['problem_type']

    # 1. Creates df_metrics DataFrame from the results stored in session state
    # First level of charts
    metrics_list = []
    for r in results:
        full_name = r.get('modelo', '')
        if '(' in full_name:
            name, opt = full_name.split('(')
            optimization = opt.replace(')', '').strip()
        else:
            name = full_name
            optimization = 'Default'
        model_name = name.strip()

        for metric, val in r.items():
            if metric in ('modelo', 'predicted_values', 'real_values'):
                continue
            metrics_list.append({
                'Model': model_name,
                'Optimization': optimization,
                'Metric': metric,
                'Value': val
            })
    df_metrics = pd.DataFrame(metrics_list)

    # 2. Shows a bar chart of AUC (classification)
    # or RMSE (regression) by optimization strategy
    st.subheader("Performance Breakdown by Optimization Strategy")

    # Metric based on problem type
    metric_field = 'roc_auc' if problem_type == 'classification' else 'RMSE'
    chart_metric = 'AUC' if metric_field == 'roc_auc' else 'RMSE'

    # Filter the metrics DataFrame for the chosen metric
    df_plot = df_metrics[df_metrics['Metric'] == metric_field]

    # Define the color scale for the bars
    color_scale = alt.Scale(
        domain=['Default', 'Genetic', 'Exhaustive'],
        range=['#384B70', '#B8001F', '#507687']
    )

    # Bar chart attributes
    bars = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('Model:N', title='Model',
                axis=alt.Axis(labelAngle=0, labelFontSize=16, titleFontSize=16)),
        y=alt.Y('Value:Q', title=chart_metric,
                axis=alt.Axis(labels=False, ticks=False, titleFontSize=18)),
        color=alt.Color('Optimization:N', scale=color_scale,
                        legend=alt.Legend(title="Parameters")),
        xOffset='Optimization:N',
        tooltip=['Model', 'Optimization', 'Value']
    )

    # Text labels attributes
    text = alt.Chart(df_plot).mark_text(
        align='center', baseline='bottom', dy=-5, fontSize=13
    ).encode(
        x='Model:N',
        xOffset='Optimization:N',
        y='Value:Q',
        text=alt.Text('Value:Q', format=".3f"),
        tooltip=['Model', 'Optimization', 'Value']
    )

    # Combine bars and text and display the chart
    chart = (bars + text).properties(width=900, height=400)
    st.altair_chart(chart, use_container_width=True)

    # Find the best model based on the same metric
    if metric_field == 'roc_auc':
        # For classification, pick the highest value
        best_idx = df_plot['Value'].idxmax()
    else:
        # For regression, pick the lowest value
        best_idx = df_plot['Value'].idxmin()

    best_row = df_plot.loc[best_idx]
    best_model, best_opt, best_val = best_row['Model'], best_row['Optimization'], best_row['Value']
    good_name = f"{best_model} ({best_opt})"
    best_data = next(r for r in results if r['modelo'] == good_name)


    #3. Shows a summary of the best model's key metrics using only exhaustive optimization

    # Determine best based on AUC (classification) or RMSE (regression)
    if 'roc_auc' in df_metrics['Metric'].values and problem_type == 'classification':
        df_auc = df_metrics[df_metrics['Metric'] == 'roc_auc']
        idx = df_auc['Value'].idxmax()
        best = df_auc.loc[idx]
        best_model = best['Model']
        best_opt = best['Optimization']
        best_val = best['Value']
        good_name = f"{best_model} ({best_opt})"
        best_data = next((r for r in results if r.get('modelo') == good_name), None)
        chart_metric = 'AUC'
    else:
        df_rmse = df_metrics[df_metrics['Metric'] == 'RMSE']
        idx = df_rmse['Value'].idxmin()
        best = df_rmse.loc[idx]
        best_model = best['Model']
        best_opt = best['Optimization']
        best_val = best['Value']
        good_name = f"{best_model} ({best_opt})"
        best_data = next((r for r in results if r.get('modelo') == good_name), None)
        chart_metric = 'RMSE'

    # Second level of charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Summary: {best_model}")
        a, b = st.columns(2)
        c, d = st.columns(2)

        if chart_metric == 'AUC' and best_data:
            a.metric("AUC", round(best_val, 3), border=True)
            b.metric("Accuracy", round(best_data.get("accuracy", 0), 3), border=True)
            c.metric("Precision", round(best_data.get("precision", 0), 3), border=True)
            d.metric("F1 Score", round(best_data.get("f1_score", 0), 3), border=True)
            st.caption("*The best algorithm was selected using AUC")
        elif chart_metric == 'RMSE' and best_data:
            a.metric("RMSE", round(best_val, 3), border=True)
            b.metric("MSE", round(best_data.get("MSE", 0), 3), border=True)
            c.metric("R2", round(best_data.get("R2", 0), 3), border=True)
            d.metric("MAE", round(best_data.get("MAE", 0), 3), border=True)
            st.caption("*The best algorithm was selected using RMSE")
        else:
            st.warning("No best model metrics found.")

    with col2:
        st.subheader("Performance Metrics by Model")
        modelo = st.session_state['modelo_supervisado']
        df_ex = modelo.get_exhaustive_metrics()
        if not df_ex.empty:
            styled = df_ex.style.apply(
                lambda s: ['background-color: #c2c7d8' if v == s.max() else '' for v in s],
                axis=0, subset=df_ex.columns[1:]
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.warning("No exhaustive results available.")
        st.caption("Results obtained using only exhaustive search.")

    # 4. Shows a confusion matrix or comparison of predicted vs actual values depending on the problem type
    # Third level of charts
    col1, col2 = st.columns(2)
    with col1:
        if problem_type == 'classification':
            st.subheader("Confusion Matrix")
            # Show confusion matrix for classification
            if best_data and 'real_values' in best_data and 'predicted_values' in best_data:
                real = best_data['real_values']
                pred = best_data['predicted_values']
                labels = sorted(set(real + pred))
                cm = confusion_matrix(real, pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                st.caption("Evaluation based on the best classification model.")
            else:
                st.warning("No confusion matrix data available.")
        else:
            st.subheader("Prediction vs Actual")
            # Choose how many points to show (no numeric labels)
            max_points = len(best_data.get('real_values', []))
            num_points = st.slider(
                "How many real value points do you want to show?",
                min_value=5,
                max_value=max_points,
                value=min(50, max_points),
                step=1,
                format=""
            )
            # Slice the data
            real = best_data['real_values'][:num_points]
            pred = best_data['predicted_values'][:num_points]
            # Build a DataFrame for line_chart
            df_line = pd.DataFrame({
                'Actual': real,
                'Predicted': pred
            })
            # Plot both series with st.line_chart
            st.line_chart(df_line, use_container_width=True, color=['#B8001F', '#384B70'])

    # 5. Shows time exacution on the optimization and algorithm execution
    with col2:
        st.subheader("Optimization Execution Time")
        # Data Optimization timings
        with open(os.path.join('dataset', 'DO_execution_time.json')) as f:
            opt_times = json.load(f)
        # Model execution timings
        with open(os.path.join('dataset', 'DM_execution_time.json')) as f:
            model_times = json.load(f)

        select = st.selectbox(
            "Select a chart to view:",
            ("Data Optimization Time", "Algorithm Execution Time")
        )

        if select == "Data Optimization Time":
            opt = opt_times.copy()
            total = opt.pop('Optimization', None)
            df_time = pd.DataFrame({
                'Optimization': list(opt.keys()),
                'Time': list(opt.values())
            })
            df_time['Optimization'] = df_time['Optimization'].str.capitalize()

            bars = alt.Chart(df_time).mark_bar().encode(
                x=alt.X('Optimization:N', axis=alt.Axis(labelAngle=0, labelFontSize=16, title=None)),
                y=alt.Y('Time:Q', axis=alt.Axis(title='Time (seconds)', titleFontSize=18)),
                color=alt.Color('Optimization:N', scale=alt.Scale(domain=['Genetic', 'Exhaustive'], range=['#B8001F', '#507687']), legend=None),
                tooltip=['Optimization', 'Time']
            )
            text = alt.Chart(df_time).mark_text(
                align='center', baseline='bottom', dy=-5, fontSize=13, fontWeight='bold'
            ).encode(
                x='Optimization:N', y='Time:Q', text=alt.Text('Time:Q', format=".2f"), tooltip=['Optimization', 'Time']
            )
            st.altair_chart((bars + text).properties(height=400), use_container_width=True)
            if total:
                st.metric("Total Optimization Processing Time", f"{total:.2f} seconds")
            st.caption("Execution time for each optimization strategy.")
        else:
            alg = model_times.copy()
            tot = alg.pop('Total', None)
            df_alg = pd.DataFrame({
                'Algorithm': list(alg.keys()),
                'Time': list(alg.values())
            }).sort_values('Time', ascending=False)
            color_set = ['#384B70', '#B8001F', '#3F51B5', '#C9B27C', '#768750', '#001FB8', '#00BCD4', '#4CAF50']

            bars = alt.Chart(df_alg).mark_bar().encode(
                x=alt.X('Algorithm:N', axis=alt.Axis(labelAngle=0, labelFontSize=14, title=None)),
                y=alt.Y('Time:Q', axis=alt.Axis(title='Time (seconds)', titleFontSize=18)),
                color=alt.Color('Algorithm:N', scale=alt.Scale(range=color_set[:len(df_alg)]), legend=None),
                tooltip=['Algorithm', 'Time']
            )
            text = alt.Chart(df_alg).mark_text(
                align='center', baseline='bottom', dy=-5, fontSize=13, fontWeight='bold'
            ).encode(
                x='Algorithm:N', y='Time:Q', text=alt.Text('Time:Q', format=".2f"), tooltip=['Algorithm', 'Time']
            )
            st.altair_chart((bars + text).properties(height=400), use_container_width=True)
            if tot:
                st.metric("Supervised Learning Processing Time", f"{tot:.2f} seconds")
            st.caption("Execution time for each algorithm.")

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

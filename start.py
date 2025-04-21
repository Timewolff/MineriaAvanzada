import streamlit as st
import model
import pandas as pd

def show():
    st.title("Let's dive into your data!")

    # 1. Upload CSV file
    st.header("1. Add your CSV file")
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

    if uploaded_file is None:
        st.warning("Please upload a CSV file to continue.")
        return

    dataset_path = uploaded_file
    st.success("Dataset loaded successfully!")

    # 2. Problem type and optimization method
    st.header("2. What do you want to do?")
    col1, col2 = st.columns(2)
    with col1:
        problem_type = st.selectbox("Problem type", ["classification", "regression", "forecast"], index=0)
    with col2:
        temp_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset for reuse
        columns = temp_df.columns.tolist()
        target_column = st.selectbox("Target column", columns, index=0)    

    # 4. Training set size
    train_size_percent = st.slider("Size of training set (%)", 50, 90, 80)
    test_size = 1 - (train_size_percent / 100.0)

    # 5. Run model
    method = "both"
    if st.button("Run Model", type="primary"):
        with st.spinner("Wait for it...", show_time=True):
            try:
                if problem_type == "forecast":
                    ft_generator = model.FakeTimeGenerator()
                    forecast_df = ft_generator.add_fake_dates(temp_df.copy(), sort_dates=True)
                    st.session_state['forecast_df'] = forecast_df
                    st.session_state['date_col'] = ft_generator.column_name
                    st.session_state['value_col'] = target_column
                    st.session_state['problem_type'] = problem_type
                    st.success("Forecast model ready to run. Go to 'Results' to see forecast.")
                    return

                eda = model.EDA(file=dataset_path)
                optimizator = model.DataOptimization(eda)
                best_params = optimizator.opti_director(
                    target_column=target_column,
                    problem_type=problem_type,
                    method=method,
                    test_size=test_size
                )
                supervised_model = model.Supervisado(optimizator, best_params=best_params)
                supervised_results = supervised_model.model_director(compare_params=True)

                st.session_state['eda'] = eda
                st.session_state['optimizador'] = optimizator
                st.session_state['best_params'] = best_params
                st.session_state['modelo_supervisado'] = supervised_model
                st.session_state['resultados'] = supervised_results
                st.session_state['problem_type'] = problem_type

                st.success("Model executed successfully. Click on 'Results' in the menu to see the report.")
            except Exception as e:
                st.error(f"Error in model execution: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    show()
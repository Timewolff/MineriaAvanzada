import streamlit as st
import os
import model
import results as results
import pandas as pd

def show():
    st.title("Let's dive into your data!")
    
    # Sección para cargar el dataset
    st.header("1. Add your CSV file")
    
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    
    use_default_dataset = st.checkbox("Note: Just for quick testing, default csv(diabetes_V2.csv)", value=True)
    
    if uploaded_file is not None:
        dataset_path = uploaded_file
        use_default_dataset = False
        st.success("Dataset loaded successfully!")
    elif use_default_dataset:
        dataset_path = os.path.join("dataset", "diabetes_V2.csv")
        if not os.path.exists(dataset_path):
            st.warning(f"The default file does not exist at {dataset_path}. Please upload a CSV file.")
            return
        st.info(f"Using default dataset: dataset\\diabetes_V2.csv")
    else:
        st.warning("Please upload a CSV file or select the option to use the default dataset.")
        return

    
    # Settings the problem type and optimization method to use
    st.header("2. What do you want to do?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        problem_type = st.selectbox(
            "Problem type",
            ["classification", "regression", "forecast"], 
            index=0
        )
        
    with col2:
        method = st.selectbox(
            "What type of optimization do you want to use?",
            ["genetic", "exhaustive", "both"],
            index=2
        )

    
    # Select the target column with the available columns
    if uploaded_file is not None:
        temp_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        columns = temp_df.columns.tolist()
        target_column = st.selectbox("Target column", columns, index=0)
    elif use_default_dataset and os.path.exists(dataset_path):
        temp_df = pd.read_csv(dataset_path)
        columns = temp_df.columns.tolist()
        target_column = st.selectbox("Target column", columns, index=0)
    else:
        st.error("Please upload a CSV file or select the default dataset first.")
        return
    
    # Slider to get the percentage of the training set
    train_size_percent = st.slider("Size of training set (%)", 50, 90, 80)
    test_size = 1 - (train_size_percent / 100)
    
    # Button to run the model
    if st.button("Run Model", type="primary"):
        try:
            if problem_type == "forecast":
                # Guardamos info necesaria para el forecast
                st.session_state['forecast_df'] = temp_df
                st.session_state['date_col'] = columns[0]  # puedes cambiar esto si necesitas una selección más específica
                st.session_state['value_col'] = target_column
                st.session_state['problem_type'] = problem_type
                st.success("Forecast model ready to run. Go to 'Results' to see forecast.")
                return

            # Carga de archivo y ejecución normal para clasificación o regresión
            eda = model.EDA(file=dataset_path)
            
            # Call the data optimization class
            optimizador = model.DataOptimization(eda)
            
            # Get the best parameters
            best_params = optimizador.opti_director(
                target_column=target_column,
                problem_type=problem_type,
                method=method,
                test_size=test_size
            )
            
            # Creation of the supervised model
            modelo_supervisado = model.Supervisado(optimizador, best_params=best_params)
            
            # Compare the results of the models
            resultados = modelo_supervisado.model_director(compare_params=True)
            
            # Store results
            st.session_state['eda'] = eda
            st.session_state['optimizador'] = optimizador
            st.session_state['best_params'] = best_params
            st.session_state['modelo_supervisado'] = modelo_supervisado
            st.session_state['resultados'] = resultados
            st.session_state['problem_type'] = problem_type
            
            # Message of success 
            st.success("Model executed successfully. Press 'Results' in the menu to see the results.")
            
        except Exception as e:
            st.error(f"Error in model execution: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    show()

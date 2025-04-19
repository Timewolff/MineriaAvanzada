import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class EDA:
    def __init__(self, file=None, delimiter=None):
        """
        Inicializa la clase EDA y carga datos desde un archivo CSV si se proporciona.

        Parámetros:
            file (str): Ruta al archivo CSV. Si no se proporciona, se inicializa un DataFrame vacío.
        """
        #self.__df = pd.read_csv(file) if file else pd.DataFrame()
        if delimiter:
            self.__df = pd.read_csv(file, delimiter=delimiter)
        else:
            self.__df = pd.read_csv(file) if file else pd.DataFrame()

    def missing_values_info(self):
        """Muestra el número de valores nulos por columna."""
        return self.__df.isnull().sum() if not self._df.empty else "No se cargaron los datos :("

    def head_df(self, n=5):
        """Muestra las primeras `n` filas del DataFrame."""
        return self.__df.head(n) if not self.__df.empty else "No se cargaron los datos :("

    def tail_df(self, n=5):
        """Muestra las últimas `n` filas del DataFrame."""
        return self.__df.tail(n) if not self.__df.empty else "No se cargaron los datos :("

    def check_data_types(self):
        """Devuelve los tipos de datos de cada columna."""
        return self.__df.dtypes

    def drop_irrelevant_columns(self, columns):
        """Elimina columnas irrelevantes del DataFrame."""
        self.__df.drop(columns=columns, inplace=True)

    def drop_missing_values(self):
        """Elimina filas con valores nulos."""
        self.__df.dropna(inplace=True)

    def detect_outliers(self):
        """Detecta valores atípicos usando el método IQR."""
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return "No hay columnas numéricas en el DataFrame."

        Q1 = num_df.quantile(0.25)
        Q3 = num_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).sum()
        Dicc_outliers = {col: outliers[col] for col in num_df.columns if outliers[col] > 0}

        return Dicc_outliers if Dicc_outliers else "No se detectaron valores atípicos en las columnas numéricas."

    def plot_scatter(self, col1, col2):
        """Generates a scatter plot for the selected columns."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.__df[col1], y=self.__df[col2])
        plt.title(f'Gráfico de Dispersión: {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid()
        st.pyplot(plt)

    def plot_histogram(self, col):
        """Generates a histogram for the selected column."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.__df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

    def plot_heatmap(self):
        """Generates a correlation heatmap for the numerical variables"""
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return "There are no numerical columns to generate the heat map"

        plt.figure(figsize=(12, 10))
        sns.heatmap(num_df.corr(), cmap="crest", annot=True, linewidths=0.5, cbar=True)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt)

    def plot_bar(self, col):
        """Genera un gráfico de barras para una columna seleccionada."""
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.__df[col])
        plt.title(f'Gráfico de Barras: {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

    def plot_violin(self, col):
        """Genera un gráfico de violín para una columna seleccionada."""
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=self.__df[col])
        plt.title(f'Gráfico de Violín: {col}')
        plt.xlabel(col)
        st.pyplot(plt)

    def plot_line(self, col):
        """Genera un gráfico de líneas para una columna seleccionada."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.__df, x=self.__df.index, y=self.__df[col])
        plt.title(f'Gráfico de Líneas: {col}')
        plt.xlabel('Índice')
        plt.ylabel(col)
        st.pyplot(plt)


    def plot_pairplot(self):
        """Genera un pairplot de las columnas numéricas."""
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return "No hay columnas numéricas para generar el pairplot."

        plt.figure(figsize=(12, 10))
        sns.pairplot(num_df)
        plt.title("Pairplot de Variables Numéricas")
        st.pyplot(plt)

    def plot_boxplot(self, col):
        """Genera un boxplot para una columna seleccionada."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.__df[col])
        plt.title(f'Boxplot de {col}')
        plt.xlabel(col)
        st.pyplot(plt)

    def __str__(self):
        return f"Clase EDA - DataFrame de la forma: {self.__df.shape}"

    def get_df(self):
        """Devuelve una copia del DataFrame para análisis posteriores."""
        return self.__df.copy()

# Visulización en Streamlit
class EDAApp:
    @staticmethod
    def show():
        st.title("Exploración de Datos - EDA")
        
        # Check if the EDA instance is available in session state
        if 'eda' not in st.session_state:
            st.warning("No data available. Please upload and process your dataset first from the 'Start Analysis' page.")
            return
        
        # Use the EDA instance from session state
        eda_instance = st.session_state['eda']
        
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

        st.subheader("Extract from the dataset")
        tab1, tab2, tab3 = st.tabs(["Top Rows", "Last Rows", "Data Types"])

        with tab1:
            st.subheader("First 5 rows of the Dataset")
            st.write(eda_instance.head_df())
        with tab2:
            st.subheader("Last 5 rows of the Dataset")
            st.write(eda_instance.tail_df())  # Fixed to use tail_df instead of head_df
        with tab3:
            st.subheader("Data types ")
            st.write(eda_instance.check_data_types())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detected Outliers")
            # Convert dictionary to DataFrame for a structured table format
            outliers_dict = eda_instance.detect_outliers() 
            if isinstance(outliers_dict, dict):
                df_outliers = pd.DataFrame(outliers_dict.items(), columns=["Feature", "Outlier Count"])
                # Display a table with the outliers
                st.table(df_outliers)
            else:
                st.write(outliers_dict)  # Display message if not a dictionary
                
        with col2:
            columna = st.selectbox("Selects a column to display the histogram", eda_instance.get_df().columns)
            eda_instance.plot_histogram(columna)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Scatter Plot")
            # Multi-select dropdown for scatter plot variables
            selected_vars = st.multiselect(
                "Select two variables", 
                eda_instance.get_df().columns, 
                default=eda_instance.get_df().columns[:2].tolist() if len(eda_instance.get_df().columns) >= 2 else [],  # Pre-select first two columns
                max_selections=2  # Limit selection to 2 variables
            )

            # Ensure only two variables are selected before plotting
            if len(selected_vars) == 2:
                eda_instance.plot_scatter(selected_vars[0], selected_vars[1])
            else:
                st.warning("Please select exactly 2 variables.")

        with col2:
            st.subheader("Heatmap")
            eda_instance.plot_heatmap()
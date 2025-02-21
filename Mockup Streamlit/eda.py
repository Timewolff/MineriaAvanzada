import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class EDA:
    def __init__(self, file=None):
        """
        Inicializa la clase EDA y carga datos desde un archivo CSV si se proporciona.

        Parámetros:
            file (str): Ruta al archivo CSV. Si no se proporciona, se inicializa un DataFrame vacío.
        """
        self.__df = pd.read_csv(file) if file else pd.DataFrame()

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
        """Genera un gráfico de dispersión entre dos columnas."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.__df[col1], y=self.__df[col2])
        plt.title(f'Gráfico de Dispersión: {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid()
        st.pyplot(plt)

    def plot_histogram(self, col):
        """Genera un histograma de la columna seleccionada."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.__df[col], kde=True)
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

    def plot_heatmap(self):
        """Genera un heatmap de correlación para las variables numéricas."""
        num_df = self.__df.select_dtypes(include=['float64', 'int64'])
        if num_df.empty:
            return "No hay columnas numéricas para generar el mapa de calor."

        plt.figure(figsize=(12, 10))
        sns.heatmap(num_df.corr(), cmap="coolwarm", annot=True, linewidths=0.5, cbar=True)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title("Correlation Heatmap", fontsize=18)
        st.pyplot(plt)

    def __str__(self):
        return f"Clase EDA - DataFrame de la forma: {self.__df.shape}"

    def get_df(self):
        """Devuelve una copia del DataFrame para análisis posteriores."""
        return self.__df.copy()


# ModelBoard: An Interactive Data Exploration and Machine Learning Tool


## 🚗 What is ModelBoard? 

ModelBoard is an easy-to-use web application designed to help users explore datasets and analyze machine learning models without requiring deep technical expertise. It provides intuitive visualizations and metrics to compare different models, making data-driven decision-making more accessible.

## Why Use ModelBoard?

📊 Easy Data Exploration: Gain insights from your data with just a few clicks.

🤖 Model Evaluation: Compare multiple regression models to find the best one for your needs.

🎨 Interactive Visualizations: Understand data trends through charts, heatmaps, and scatter plots.

🚀 No Coding Required: Designed for data analysts, business users, and researchers.

---
## How It Works

Upload Your Data: Simply load your dataset in CSV format.

Explore Your Data: View statistics, missing values, and correlations.

Analyze Models: Compare regression models based on performance metrics.

Visualize Results: Generate charts for deeper insights.

---
## Features

Exploratory Data Analysis (EDA): Identify missing values, detect outliers, and understand distributions.

Regression Model Comparison: Evaluate different machine learning models using R², RMSE, and MAE.

Data Visualization: Generate interactive plots, correlation heatmaps, and predictive trend charts.

Simple UI: A user-friendly interface powered by Streamlit.
---

## Installation

To run ModelBoard on your local machine:
pip install -r requirements.txt
streamlit run app.py

---
## **Project Structure**
```
📂 ModelBoard/              # Main project folder
│── 📂 modules/             # Core application modules
│   ├── eda.py             # Exploratory Data Analysis module
│   ├── results.py         # Model Evaluation and visualization
│   ├── about.py           # Project Information and Contributors
│── 📂 data/               # Dataset storage
│   ├── sample_data.csv    # Example dataset
│── 📂 config/             # Configuration settings
│   ├── config.json        # Model parameters and settings
│── 📄 app.py              # Main Streamlit application
│── 📄 requirements.txt    # Dependencies
│── 📄 README.md           # Documentation
```

## **Technologies Used**

Python 3.8+
Streamlit for interactive web applications
Pandas & NumPy for data manipulation
Seaborn & Matplotlib for visualization
Altair for interactive charts
Streamlit-AntD-Components for enhanced UI

---
## **Future Enhancements**

✅ More model evaluation techniques.
✅ Additional interactive visualizations.
✅ Improved UI and usability.
✅ Integration of advanced ML algorithms.

---
## **Contributors**

- Carolina Salas Moreno
- Deykel Bernard Salazar
- Esteban Ramirez Montano
- Kristhel Porras Mata
- Marla Gomez Hernández

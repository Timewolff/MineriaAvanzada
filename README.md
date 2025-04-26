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

## Requirements
**Step 1:** Please install Microsoft C++ Build Tools in your machine. 
**Step 2:** Install Python 3.11.7
**Step 3:** Run the following code if this is your first time running it `pip install -r requirements.txt`

---
## **Project Structure**
```
📂 MineriaApp/                     # Main project folder
│
├── 📂 dataset/                   # CSV datasets tested in the project
│   ├── dataset.csv
│   ├── diabetes_V2.csv
│   ├── expenses.csv
│   └── potabilidad_V2.csv
│
├── 📂 notebooks/                 # Jupyter Notebooks (experimentation)
│
├── 📂 runtime_json_files/        # JSON files for tracking runtime execution
│   ├── DM_execution_time.json
│   └── DO_execution_time.json
│
├── 📂 .streamlit/                # Streamlit configuration folder
│
├── 📄 .gitignore                 # Git ignored files and folders
├── 📄 about.py                   # Project details
├── 📄 app.py                     # Main Streamlit application entry point
├── 📄 eda.py                     # Exploratory Data Analysis module
├── 📄 model.py                   # Model training and evaluation
├── 📄 requirements.txt           # Project dependencies
├── 📄 results.py                 # Model results and visualizations
├── 📄 start.py                   # Initial Analysis module
└── 📄 README.md                  # Project documentation
```

## **Technologies Used**

Python 3.8+
Streamlit for interactive web applications
Pandas & NumPy for data manipulation
Seaborn & Matplotlib for visualization
Altair for interactive charts
Streamlit-AntD-Components for enhanced UI

---
## **Contributors**

- Carolina Salas Moreno
- Deykel Bernard Salazar
- Esteban Ramirez Montano
- Kristhel Porras Mata
- Marla Gomez Hernández

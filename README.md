# 💠 Telecom | Advanced Customer Churn Prediction System

An end-to-end Machine Learning pipeline and interactive CRM dashboard designed to predict customer churn, identify key risk factors, segment customer bases, and provide **Explainable AI** insights using SHAP and advanced visualizations.

## ✨ Features

- **Predictive Modeling**: Utilizes Scikit-Learn and XGBoost models to accurately forecast the probability of a customer churning based on demographic, service, and billing data.
- **Explainable AI (XAI)**: Demystifies "black-box" models by implementing SHAP (SHapley Additive exPlanations) to dynamically generate localized waterfall charts, explaining exactly *why* a specific customer is predicted to churn.
- **Customer Segmentation**: Employs K-Means Clustering on customer lifetime value (LTV) and tenure data to segment users into actionable groups automatically (e.g., High Value VIPs, At-Risk New Users).
- **Interactive CRM Dashboard**: Built with Streamlit and Plotly, the dashboard features a custom sleek UI with animated risk dials, sidebar inputs, and multi-tab global data overviews.
- **Business Intelligence Ready**: Seamless script to export consolidated insights and model predictions directly into a flattened CSV structured explicitly for Power BI.

## 🛠️ Technology Stack

- **Data Processing & ML Core**: `pandas`, `numpy`, `scikit-learn`, `xgboost`
- **Model Explainability**: `shap`
- **Web App & UI Engine**: `streamlit`, `plotly`
- **Serialization**: `joblib`

## 📂 Project Structure

```text
churn_prediction_system/
│
├── app.py                          # The main Streamlit Dashboard UI
├── generate_data.py                # Synthetic dataset generation logic
├── requirements.txt                # Python dependencies
│
├── data/                           
│   ├── raw/                        # Original unprocessed dataset
│   ├── processed/                  # Cleaned and encoded data ready for modeling
│   ├── eda_plots/                  # Pre-rendered Exploratory Data Analysis images
│   └── power_bi/                   # Flattened exports for Business Intelligence
│
├── models/                         # Serialized Pipeline Assets
│   ├── best_model.pkl              # Primary classification model
│   ├── scaler.pkl                  # Feature normalizer
│   ├── kmeans_segmentation.pkl     # Unsupervised clustering model
│   ├── shap_background.pkl         # Reference data for linear explainers
│   └── model_columns.pkl           # Feature alignment registry
│
├── notebooks/                      # Jupyter workspace
│   └── Churn_Analysis.ipynb        # Interactive exploration notebook
│
└── src/                            # Core Pipeline Source Code
    ├── data_preprocessing.py       # Cleaning, outlier handling, encoding
    ├── eda.py                      # Data visualizations & statistics
    ├── segmentation.py             # K-Means clustering pipeline
    ├── modeling.py                 # Training XGB/RF/LogReg, hyperparameter tuning
    ├── explainability.py           # SHAP summary generation
    └── export_for_bi.py            # Consolidation logic merging predictions
```

## 🚀 Setup & Installation

**1. Set up a virtual environment**
- **Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate data & train models**
If you are starting entirely from scratch, you can run the pipeline sequentially to generate synthetic data and serialize newly trained AI models into the `models/` folder:
```bash
python generate_data.py
python src/data_preprocessing.py
python src/segmentation.py
python src/modeling.py
```

## 💻 Running the Dashboard locally
To launch the interactive CRM interface, ensure your virtual environment is active and run:
```bash
streamlit run app.py
```
*The app should automatically open in your browser at `http://localhost:8501`.* 

## 📊 Exporting to Power BI
To update the Business Intelligence dataset utilizing the latest model scoring, run:
```bash
python src/export_for_bi.py
```
This will generate `PowerBI_Customer_Insights.csv` in the `data/power_bi/` folder, which can be directly loaded into Power BI or Tableau.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Built with ❤️ utilizing Scikit-Learn, Streamlit, Plotly, and SHAP.*

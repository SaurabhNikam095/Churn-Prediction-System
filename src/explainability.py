import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def generate_shap_explanations(data_path="data/processed/engineered_dataset.csv", model_path="models/best_model.pkl", output_dir="data/eda_plots"):
    """
    Generates SHAP summary plots to explain the best model's predictions.
    """
    print("Generating SHAP Explanations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    model_obj = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)
    
    # We sample the data to speed up SHAP calculation for the global summary plot
    # 500 samples is usually sufficient for a good global representation without taking too long
    X_sample = X.sample(n=min(500, len(X)), random_state=42)
    
    # Create the SHAP Explainer
    # TreeExplainer is much faster and perfect for Random Forest and XGBoost
    if type(model_obj).__name__ in ['RandomForestClassifier', 'XGBClassifier', 'XGBRegressor']:
        explainer = shap.TreeExplainer(model_obj)
        # XGBoost output shape handling
        shap_values = explainer.shap_values(X_sample)
        
        # Random Forest shap_values returns a list for each class.
        # XGBoost usually returns a single array for binary classification.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Take the 'Churn=Yes' class
            
    else:
        # For Logistic Regression or others, use a generic explainer (KernelExplainer)
        # using the background data we saved earlier
        background_data = joblib.load("models/shap_background.pkl")
        explainer = shap.KernelExplainer(model_obj.predict_proba, background_data)
        
        # get shap values for class 1 (Churn)
        shap_values_full = explainer.shap_values(X_sample)
        if isinstance(shap_values_full, list):
            shap_values = shap_values_full[1]
        else:
            shap_values = shap_values_full
            
    # 1. Generate Global Summary Plot (Bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    # Adjust layout so feature names aren't cut off
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_bar.png")
    plt.close()
    
    # 2. Generate Global Summary Plot (Dot/Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Global Feature Impact (SHAP Beeswarm)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_dot.png")
    plt.close()
    
    print(f"SHAP plots successfully generated and saved to {output_dir}")

if __name__ == "__main__":
    generate_shap_explanations()

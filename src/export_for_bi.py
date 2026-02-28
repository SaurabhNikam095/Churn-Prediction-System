import pandas as pd
import joblib
import os

def export_for_powerbi(data_path="data/raw/customer_churn_dataset.csv", engineered_path="data/processed/engineered_dataset.csv", model_path="models/best_model.pkl", output_dir="data/power_bi"):
    """
    Generates a comprehensive dataset for Power BI by merging original inputs with model predictions, probabilities, and segments.
    """
    print("Exporting data for Power BI Dashboard...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    raw_df = pd.read_csv(data_path)
    eng_df = pd.read_csv(engineered_path)
    
    # 2. Load Models
    model = joblib.load(model_path)
    kmeans = joblib.load("models/kmeans_segmentation.pkl")
    kmeans_scaler = joblib.load("models/kmeans_scaler.pkl")
    
    X_pred = eng_df.drop('Churn', axis=1)
    
    # 3. Predict Churn Probability
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)[:, 1] # Probability of Churn
    
    # 4. Predict Segments
    # Use original columns that were used for tracking segment
    segment_features = raw_df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    segment_features['TotalCharges'] = pd.to_numeric(segment_features['TotalCharges'], errors='coerce')
    segment_features['TotalCharges'].fillna(segment_features['MonthlyCharges'], inplace=True)
    
    scaled_seg_fw = kmeans_scaler.transform(segment_features)
    segments = kmeans.predict(scaled_seg_fw)
    
    segment_names = {0: "Low/Mid Value (Needs Nurturing)", 1: "High Value (VIP)", 2: "New/At Risk (Monitor)"}
    segment_labels = [segment_names[s] for s in segments]
    
    # 5. Merge all into a clean, comprehensive BI dataset
    bi_df = raw_df.copy()
    bi_df['Predicted_Churn_Risk'] = probabilities.round(4)
    bi_df['Predicted_Churn_Class'] = ['Yes' if p == 1 else 'No' for p in predictions]
    bi_df['Customer_Segment'] = segment_labels
    
    # 6. Save for Power BI
    output_filepath = os.path.join(output_dir, "PowerBI_Customer_Insights.csv")
    bi_df.to_csv(output_filepath, index=False)
    
    print(f"Power BI Dataset generated successfully: {output_filepath}")

if __name__ == "__main__":
    export_for_powerbi()

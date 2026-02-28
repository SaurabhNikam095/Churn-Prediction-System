import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def feature_engineering(filepath="data/processed/cleaned_dataset.csv", output_path="data/processed/engineered_dataset.csv"):
    """
    Encodes categorical variables and scales numerical features.
    """
    print("Starting Feature Engineering...")
    df = pd.read_csv(filepath)
    
    # 1. Drop customerID as it's not a predictive feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # 2. Identify column types
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove Target variable from processing lists
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    if 'Churn_Num' in numerical_cols: # Cohort analysis artifact
        numerical_cols.remove('Churn_Num')
        df = df.drop('Churn_Num', axis=1, errors='ignore')
    if 'TenureCohort' in categorical_cols:
        categorical_cols.remove('TenureCohort')
        df = df.drop('TenureCohort', axis=1, errors='ignore')

    # 3. Encode Categorical Variables (Label Encoding for binary, One-Hot for multi-class)
    print("Encoding categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        if df[col].nunique() <= 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # One-Hot Encoding for remaining >2 class variables
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c not in label_encoders.keys()], drop_first=True)
    
    # Encode target variable
    target_le = LabelEncoder()
    df['Churn'] = target_le.fit_transform(df['Churn'])
    
    # 4. Scale Numerical Features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    # Ensure we only scale actual numerical columns present in the dataframe
    num_cols_to_scale = [col for col in numerical_cols if col in df.columns]
    df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
    
    # 5. Save artifacts for later use (app deployment)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    # Save a generic dictionary of columns to reconstruct the input dataframe shape later
    joblib.dump(list(df.columns), "models/model_columns.pkl")
    
    # Save the engineered dataset
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    feature_engineering()

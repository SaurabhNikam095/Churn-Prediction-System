import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_split_data(filepath="data/processed/engineered_dataset.csv"):
    df = pd.read_csv(filepath)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 80-20 split with stratification to maintain churn ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dataset split: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

def plot_confusion_matrix(cm, model_name, output_dir="data/model_evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{output_dir}/cm_{model_name.replace(' ', '_').lower()}.png")
    plt.close()

def run_modeling_pipeline():
    print("Starting Predictive Modeling phase...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_roc_auc = 0
    best_model_obj = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # XGBoost and RF have predict_proba by default. LR does too if configured (it is).
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics
        
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name)
        
        print(f"Metrics for {name}: ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        # We'll select the best model based on ROC-AUC
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_model_name = name
            best_model_obj = model
            
    # Save the evaluation metrics
    os.makedirs("models", exist_ok=True)
    with open("data/model_evaluation/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nTraining complete. Best model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    # Save the best model for Deployment/Explainability phases
    joblib.dump(best_model_obj, "models/best_model.pkl")
    # Save the training data sample for SHAP background
    joblib.dump(X_train.sample(min(100, len(X_train)), random_state=42), "models/shap_background.pkl")
    print(f"Saved {best_model_name} to models/best_model.pkl")

if __name__ == "__main__":
    run_modeling_pipeline()

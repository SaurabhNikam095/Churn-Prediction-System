import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def perform_segmentation(filepath="data/processed/cleaned_dataset.csv", output_dir="data/eda_plots"):
    """
    Performs Customer Segmentation based on billing/usage behaviors using KMeans.
    We'll use specifically numerical behavioral features that make business sense: MonthlyCharges and tenure.
    """
    print("Starting Customer Segmentation (KMeans)...")
    df = pd.read_csv(filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    # Select features for segmentation
    # We use the un-engineered cleaned data because we want easily interpretable clusters
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Handle any potential remaining NaNs just in case
    segment_data = df[features].fillna(0)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(segment_data)
    
    # Apply KMeans (Choosing 3 segments: e.g., New/Low Value, Loyal/Low Value, High Value)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['CustomerSegment'] = kmeans.fit_predict(scaled_data)
    
    # Save the clustering model
    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_segmentation.pkl")
    joblib.dump(scaler, "models/kmeans_scaler.pkl")
    
    # Visualize the Segments
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='CustomerSegment', palette='viridis', alpha=0.6)
    plt.title('Customer Segmentation (Tenure vs Monthly Charges)')
    plt.savefig(f"{output_dir}/customer_segments.png")
    plt.close()
    
    # Calculate Cluster Profiles for Business Understanding
    cluster_profiles = df.groupby('CustomerSegment')[features + ['Churn']].agg({
        'tenure': 'mean',
        'MonthlyCharges': 'mean',
        'TotalCharges': 'mean',
        'Churn': lambda x: (x == 'Yes').mean() * 100 # Churn %
    }).round(2)
    
    print("\nCluster Profiles (Averages & Churn Rate %):")
    print(cluster_profiles)
    
    # Save segmented data
    df.to_csv("data/processed/segmented_dataset.csv", index=False)
    print("Segmentation complete. Segmented dataset saved.")

if __name__ == "__main__":
    perform_segmentation()

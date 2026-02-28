import pandas as pd
import os

url = "https://raw.githubusercontent.com/Blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
output_path = "data/raw/customer_churn_dataset.csv"

def download_data():
    print(f"Downloading dataset from {url}...")
    try:
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset downloaded and saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Failed to download data: {e}")

if __name__ == "__main__":
    download_data()

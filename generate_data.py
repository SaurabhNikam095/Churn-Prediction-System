import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_telco_data(num_samples=7043):
    print("Generating synthetic IBM Telco Customer Churn dataset...")
    np.random.seed(42)
    random.seed(42)

    data = {
        'customerID': [f"{random.randint(1000, 9999)}-{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}" for _ in range(num_samples)],
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'SeniorCitizen': np.random.choice([0, 1], num_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], num_samples),
        'Dependents': np.random.choice(['Yes', 'No'], num_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(0, 73, num_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], num_samples, p=[0.9, 0.1])
    }
    
    # Dependent features
    multiple_lines = []
    for p in data['PhoneService']:
        if p == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(random.choice(['Yes', 'No']))
    data['MultipleLines'] = multiple_lines

    data['InternetService'] = np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples, p=[0.34, 0.44, 0.22])
    
    def internet_dependent(service, choices):
        return [random.choice(choices) if s != 'No' else 'No internet service' for s in service]

    data['OnlineSecurity'] = internet_dependent(data['InternetService'], ['Yes', 'No'])
    data['OnlineBackup'] = internet_dependent(data['InternetService'], ['Yes', 'No'])
    data['DeviceProtection'] = internet_dependent(data['InternetService'], ['Yes', 'No'])
    data['TechSupport'] = internet_dependent(data['InternetService'], ['Yes', 'No'])
    data['StreamingTV'] = internet_dependent(data['InternetService'], ['Yes', 'No'])
    data['StreamingMovies'] = internet_dependent(data['InternetService'], ['Yes', 'No'])

    data['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples, p=[0.55, 0.21, 0.24])
    data['PaperlessBilling'] = np.random.choice(['Yes', 'No'], num_samples, p=[0.59, 0.41])
    data['PaymentMethod'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_samples)

    data['MonthlyCharges'] = np.random.uniform(18.25, 118.75, num_samples).round(2)
    
    # Calculate total charges (approx tenure * monthly + some noise)
    total_charges = []
    for t, m in zip(data['tenure'], data['MonthlyCharges']):
        if t == 0:
            total_charges.append(" ")
        else:
            total_charges.append(str(round(t * m + random.uniform(-50, 50), 2)))
    data['TotalCharges'] = total_charges

    # Target class (Churn) - bias towards realistic behavior
    churn = []
    for i in range(num_samples):
        prob = 0.26 # base
        if data['Contract'][i] == 'Month-to-month': prob += 0.15
        if data['InternetService'][i] == 'Fiber optic': prob += 0.1
        if data['tenure'][i] < 12: prob += 0.1
        if data['TechSupport'][i] == 'No': prob += 0.05
        prob = min(0.95, prob)
        churn.append('Yes' if random.random() < prob else 'No')
    data['Churn'] = churn

    df = pd.DataFrame(data)
    
    output_path = "data/raw/customer_churn_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_telco_data()

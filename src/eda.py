import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(filepath="data/processed/cleaned_dataset.csv", output_dir="data/eda_plots"):
    """
    Generates Exploratory Data Analysis plots and saves them.
    """
    df = pd.read_csv(filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution')
    plt.savefig(f"{output_dir}/churn_distribution.png")
    plt.close()

    # 2. Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True)
    plt.title('Monthly Charges vs Churn')
    plt.savefig(f"{output_dir}/monthly_charges_vs_churn.png")
    plt.close()
    
    # 3. Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True)
    plt.title('Tenure vs Churn')
    plt.savefig(f"{output_dir}/tenure_vs_churn.png")
    plt.close()
    
    # 4. Correlation Matrix
    plt.figure(figsize=(10, 8))
    # Convert target to numeric temporarily for correlation
    df_corr = df.copy()
    df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
    # Select only numeric columns
    numeric_df = df_corr.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

def perform_cohort_analysis(filepath="data/processed/cleaned_dataset.csv", output_dir="data/eda_plots"):
    """
    Performs Cohort Analysis based on tenure.
    Divides customers into cohorts (e.g., 0-12 months, 13-24 months, etc.)
    and analyzes churn rates.
    """
    df = pd.read_csv(filepath)
    
    # Create tenure cohorts (bins of 12 months)
    bins = [0, 12, 24, 36, 48, 60, 72, 80]
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72', '72+']
    df['TenureCohort'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
    
    # Calculate churn rate per cohort
    df['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0})
    cohort_churn = df.groupby('TenureCohort')['Churn_Num'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=cohort_churn, x='TenureCohort', y='Churn_Num', palette='viridis')
    plt.title('Churn Rate by Tenure Cohort (Retained Months)')
    plt.ylabel('Churn Rate')
    plt.xlabel('Tenure Cohort (Months)')
    plt.savefig(f"{output_dir}/cohort_analysis.png")
    plt.close()
    
    print("Cohort analysis plot saved.")

if __name__ == "__main__":
    perform_eda()
    perform_cohort_analysis()

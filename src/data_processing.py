"""
Data Processing Utilities for Customer Churn Prediction
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def generate_sample_churn_data(n_samples=1000, random_state=42):
    """
    Generate synthetic customer churn data for demonstration purposes.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Synthetic customer data with features and churn label
    """
    np.random.seed(random_state)
    
    # Customer demographics
    age = np.random.randint(18, 80, n_samples)
    tenure_months = np.random.randint(0, 120, n_samples)
    
    # Account information
    monthly_charges = np.random.uniform(20, 200, n_samples)
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 100, n_samples)
    
    # Service features
    num_products = np.random.randint(1, 5, n_samples)
    has_online_service = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    has_tech_support = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Interaction features
    num_customer_service_calls = np.random.poisson(3, n_samples)
    num_claims_last_year = np.random.poisson(1.5, n_samples)
    
    # Customer satisfaction (hidden variable that influences churn)
    satisfaction_score = (
        0.3 * (tenure_months / 120) +
        0.2 * (1 - monthly_charges / 200) +
        0.2 * has_online_service +
        0.15 * has_tech_support +
        0.15 * (1 - np.minimum(num_customer_service_calls / 10, 1))
    ) + np.random.normal(0, 0.1, n_samples)
    
    # Churn probability based on various factors
    churn_probability = (
        0.8 - satisfaction_score +
        0.1 * (1 - tenure_months / 120) +
        0.05 * (monthly_charges / 200) +
        0.05 * (num_customer_service_calls / 10)
    )
    churn_probability = np.clip(churn_probability, 0, 1)
    
    # Generate churn labels
    churn = np.random.binomial(1, churn_probability)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'num_products': num_products,
        'has_online_service': has_online_service,
        'has_tech_support': has_tech_support,
        'num_customer_service_calls': num_customer_service_calls,
        'num_claims_last_year': num_claims_last_year,
        'churn': churn
    })
    
    return data


def preprocess_data(data, target_column='churn', test_size=0.2, random_state=42):
    """
    Preprocess data for model training.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Name of target column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def save_sample_data(output_path='data/customer_churn.csv', n_samples=1000):
    """
    Generate and save sample churn data to CSV.
    
    Args:
        output_path (str): Path to save the CSV file
        n_samples (int): Number of samples to generate
    """
    data = generate_sample_churn_data(n_samples=n_samples)
    data.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    return data


if __name__ == "__main__":
    # Generate and save sample data
    data = save_sample_data()
    print(f"\nDataset shape: {data.shape}")
    print(f"\nFeatures:\n{data.columns.tolist()}")
    print(f"\nChurn distribution:\n{data['churn'].value_counts()}")
    print(f"\nSample records:\n{data.head()}")

# Data Directory

This directory contains customer churn datasets used for training and testing the AI explainability models.

## Files
- `customer_churn.csv` - Generated sample customer churn data (auto-generated when running scripts)

## Data Schema
The customer churn dataset includes the following features:
- `age`: Customer age
- `tenure_months`: How long the customer has been with the company
- `monthly_charges`: Monthly subscription/premium charges
- `total_charges`: Total charges over the customer's lifetime
- `num_products`: Number of insurance products the customer has
- `has_online_service`: Whether customer uses online services (0/1)
- `has_tech_support`: Whether customer has technical support (0/1)
- `num_customer_service_calls`: Number of customer service calls
- `num_claims_last_year`: Number of insurance claims in the past year
- `churn`: Target variable - whether customer churned (0/1)

## Generating Data
To generate sample data, run:
```python
from src.data_processing import save_sample_data
save_sample_data('data/customer_churn.csv', n_samples=1000)
```

# Quick Start Guide - ChubbChurns AI Explainability

This guide will help you get started with the ChubbChurns AI Explainability project in minutes.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ntini773/ChubbChurns.git
cd ChubbChurns
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Demo

### Option 1: Quick Example Script
The fastest way to see AI explainability in action:

```bash
python example.py
```

This will:
- Generate sample customer churn data
- Train a Random Forest classifier
- Create SHAP and LIME explanations
- Save visualizations to the `visualizations/` directory

Expected output:
```
ChubbChurns - AI Explainability Demo
========================================================
[1/6] Generating sample customer data...
  ✓ Generated 1000 customer records
  ✓ Churn rate: 35.40%

[2/6] Preprocessing data...
  ✓ Training samples: 800
  ✓ Test samples: 200
  ✓ Features: 9
...
```

### Option 2: Interactive Jupyter Notebook
For a more interactive experience with detailed explanations:

```bash
jupyter notebook notebooks/AI_Explainability_Demo.ipynb
```

### Option 3: Use the Modules in Your Own Code
```python
from src.data_processing import generate_sample_churn_data, preprocess_data
from src.model import ChurnPredictor
from src.explainability import SHAPExplainer, LIMEExplainer

# Generate data
data = generate_sample_churn_data(n_samples=1000)
X_train, X_test, y_train, y_test, scaler, features = preprocess_data(data)

# Train model
model = ChurnPredictor(model_type='random_forest')
model.train(X_train, y_train)

# Generate explanations
shap_explainer = SHAPExplainer(model, feature_names=features)
shap_explainer.create_explainer(X_train.sample(100))
shap_values = shap_explainer.explain(X_test)
```

## Understanding the Output

### SHAP Summary Plot
- Shows global feature importance across all predictions
- Features are ranked by average impact on model output
- Color indicates feature value (red = high, blue = low)

### SHAP Waterfall Plot
- Explains a single prediction
- Shows how each feature pushes prediction from base value
- Bars point right (positive impact) or left (negative impact)

### LIME Explanation
- Local approximation of model behavior
- Shows which features contribute to a specific prediction
- Easier to understand for non-technical stakeholders

## Common Use Cases

### 1. Identify At-Risk Customers
```python
# Get high-risk customers
predictions = model.predict_proba(X_test)
high_risk_indices = predictions[:, 1] > 0.7

# Explain why they're at risk
for idx in high_risk_indices[:5]:
    explanation = lime_explainer.explain_instance(X_test.iloc[idx])
    print(f"Customer {idx}: {explanation.as_list()}")
```

### 2. Validate Model Logic
```python
# Check if model uses expected features
feature_importance = model.get_feature_importance()
print(feature_importance.head(10))

# Verify with SHAP
shap_explainer.plot_summary(X_test, save_path='validation.png')
```

### 3. Generate Reports for Stakeholders
```python
# Create explanations for business team
for i in range(5):
    lime_explainer.plot_explanation(
        lime_explainer.explain_instance(X_test.iloc[i]),
        save_path=f'reports/customer_{i}_explanation.png'
    )
```

## Troubleshooting

### Installation Issues
If you encounter dependency conflicts:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Memory Issues with SHAP
For large datasets, use a sample:
```python
sample = X_train.sample(min(100, len(X_train)))
shap_explainer.create_explainer(sample)
```

### Visualization Not Showing
Make sure matplotlib backend is set correctly:
```python
import matplotlib
matplotlib.use('Agg')  # For saving files
# or
matplotlib.use('TkAgg')  # For displaying
```

## Next Steps

1. **Explore Different Models**: Try Gradient Boosting or Logistic Regression
2. **Custom Data**: Replace sample data with your own customer data
3. **Feature Engineering**: Add domain-specific features
4. **Hyperparameter Tuning**: Optimize model performance
5. **Production Deployment**: Integrate explanations into your application

## Resources

- **SHAP Documentation**: https://shap.readthedocs.io/
- **LIME Documentation**: https://lime-ml.readthedocs.io/
- **Scikit-learn**: https://scikit-learn.org/

## Getting Help

- Check the README.md for detailed documentation
- Review example code in `example.py`
- Explore the Jupyter notebook for interactive examples
- Open an issue on GitHub for bugs or questions

---

Happy Explaining! 🎯

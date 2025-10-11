# ChubbChurns - AI Explainability for Customer Churn Prediction

## Overview
ChubbChurns is a hackathon project focused on **AI Explainability** for predicting customer churn in the insurance industry. This repository demonstrates how to build interpretable machine learning models and explain their predictions using state-of-the-art explainability techniques.

## What is AI Explainability?
AI Explainability (also known as Explainable AI or XAI) refers to methods and techniques that make the results and outputs of AI models understandable to humans. In the context of customer churn prediction, it helps answer questions like:
- Why did the model predict this customer would churn?
- Which features contributed most to this prediction?
- How can we trust and validate the model's decisions?

## Features
- **Customer Churn Prediction Model**: Machine learning model to predict customer churn
- **SHAP (SHapley Additive exPlanations)**: Unified approach to explain model predictions
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local explanations for individual predictions
- **Feature Importance Analysis**: Understand which features matter most
- **Interactive Visualizations**: Visual explanations of model behavior
- **Example Notebooks**: Step-by-step tutorials on using explainability tools

## Project Structure
```
ChubbChurns/
├── data/                  # Sample datasets
├── models/                # Trained models
├── src/                   # Source code
│   ├── data_processing.py # Data preprocessing utilities
│   ├── model.py          # Churn prediction model
│   └── explainability.py # SHAP and LIME implementations
├── notebooks/            # Jupyter notebooks with examples
├── visualizations/       # Generated explanation plots
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/ntini773/ChubbChurns.git
cd ChubbChurns

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
# Example: Generate explanations for a churn prediction
from src.model import ChurnPredictor
from src.explainability import SHAPExplainer, LIMEExplainer

# Load model and make predictions
model = ChurnPredictor()
prediction = model.predict(customer_data)

# Generate SHAP explanations
shap_explainer = SHAPExplainer(model)
shap_values = shap_explainer.explain(customer_data)

# Generate LIME explanations
lime_explainer = LIMEExplainer(model)
explanation = lime_explainer.explain_instance(customer_data)
```

## Key Explainability Techniques

### 1. SHAP (SHapley Additive exPlanations)
- Provides consistent and accurate feature importance
- Based on game theory concepts
- Works with any machine learning model
- Offers both local (instance-level) and global explanations

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions
- Model-agnostic approach
- Creates interpretable local surrogate models
- Easy to understand for non-technical stakeholders

### 3. Feature Importance
- Identifies which features drive predictions
- Helps validate business logic
- Supports model debugging and improvement

## Use Cases
1. **Customer Retention**: Identify at-risk customers and understand why they might churn
2. **Model Validation**: Ensure the model uses appropriate features and logic
3. **Regulatory Compliance**: Provide transparent explanations for model decisions
4. **Business Insights**: Discover actionable patterns in customer behavior

## Contributing
This is a hackathon project. Feel free to fork, experiment, and submit pull requests!

## License
MIT License

## Acknowledgments
- SHAP library by Scott Lundberg
- LIME library by Marco Tulio Ribeiro
- Scikit-learn community
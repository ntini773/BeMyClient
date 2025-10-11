# Visualizations Directory

This directory contains AI explainability visualizations generated from SHAP and LIME analyses.

## SHAP Visualizations
- `shap_summary.png` - Global feature importance summary
- `shap_waterfall_*.png` - Individual prediction explanations
- `shap_force_*.png` - Force plots showing feature contributions

## LIME Visualizations
- `lime_explanation_*.png` - Local explanations for individual predictions

## Generating Visualizations
Run the example script or Jupyter notebook to generate visualizations:
```bash
python example.py
```

Or use the explainability modules directly:
```python
from src.explainability import SHAPExplainer, LIMEExplainer

# Generate SHAP plots
shap_explainer.plot_summary(X_test, save_path='visualizations/shap_summary.png')

# Generate LIME plots
lime_explainer.plot_explanation(explanation, save_path='visualizations/lime_explanation.png')
```

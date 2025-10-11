# Models Directory

This directory stores trained machine learning models for customer churn prediction.

## Files
- `churn_model.pkl` - Trained churn prediction model (auto-generated when saving models)

## Model Types
The repository supports multiple model types:
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression

## Saving a Model
```python
from src.model import ChurnPredictor

model = ChurnPredictor(model_type='random_forest')
model.train(X_train, y_train)
model.save('models/churn_model.pkl')
```

## Loading a Model
```python
from src.model import ChurnPredictor

model = ChurnPredictor()
model.load('models/churn_model.pkl')
predictions = model.predict(X_test)
```

"""
Customer Churn Prediction Model
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os


class ChurnPredictor:
    """
    Customer Churn Prediction Model with multiple algorithm support.
    """
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Initialize the churn predictor.
        
        Args:
            model_type (str): Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the machine learning model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the churn prediction model.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training features
            y_train (pd.Series or np.ndarray): Training labels
            
        Returns:
            self: Trained model instance
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """
        Predict churn for given customers.
        
        Args:
            X (pd.DataFrame or np.ndarray): Customer features
            
        Returns:
            np.ndarray: Predicted churn labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict churn probability for given customers.
        
        Args:
            X (pd.DataFrame or np.ndarray): Customer features
            
        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test (pd.DataFrame or np.ndarray): Test features
            y_test (pd.Series or np.ndarray): Test labels
            
        Returns:
            dict: Dictionary containing various performance metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance scores from the model.
        
        Returns:
            pd.DataFrame: Feature names and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance.")
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importances))],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save(self, filepath='models/churn_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='models/churn_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            self: Loaded model instance
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    from data_processing import generate_sample_churn_data, preprocess_data
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_churn_data(n_samples=1000)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    # Train model
    print("\nTraining Random Forest model...")
    model = ChurnPredictor(model_type='random_forest')
    model.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Performance:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    print(model.get_feature_importance())
    
    # Save model
    model.save()

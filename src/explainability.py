"""
AI Explainability Module - SHAP and LIME implementations for churn prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
import os


class SHAPExplainer:
    """
    SHAP-based explainer for customer churn predictions.
    """
    
    def __init__(self, model, feature_names=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model with predict or predict_proba method
            feature_names (list): List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, X_background, algorithm='tree'):
        """
        Create SHAP explainer based on model type.
        
        Args:
            X_background (pd.DataFrame or np.ndarray): Background data for SHAP
            algorithm (str): Type of SHAP explainer ('tree', 'kernel', 'linear')
            
        Returns:
            self: Explainer instance
        """
        if algorithm == 'tree':
            # For tree-based models (Random Forest, Gradient Boosting)
            if hasattr(self.model, 'model'):
                self.explainer = shap.TreeExplainer(self.model.model)
            else:
                self.explainer = shap.TreeExplainer(self.model)
        elif algorithm == 'kernel':
            # Model-agnostic explainer
            if hasattr(self.model, 'predict_proba'):
                predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
            else:
                predict_fn = self.model.predict
            self.explainer = shap.KernelExplainer(predict_fn, X_background)
        elif algorithm == 'linear':
            # For linear models
            if hasattr(self.model, 'model'):
                self.explainer = shap.LinearExplainer(self.model.model, X_background)
            else:
                self.explainer = shap.LinearExplainer(self.model, X_background)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return self
    
    def explain(self, X):
        """
        Generate SHAP values for given instances.
        
        Args:
            X (pd.DataFrame or np.ndarray): Instances to explain
            
        Returns:
            np.ndarray: SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, get SHAP values for positive class
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]
        
        return self.shap_values
    
    def plot_summary(self, X, save_path=None):
        """
        Create SHAP summary plot showing feature importance.
        
        Args:
            X (pd.DataFrame or np.ndarray): Data to explain
            save_path (str): Optional path to save the plot
        """
        if self.shap_values is None:
            self.explain(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_waterfall(self, X, instance_index=0, save_path=None):
        """
        Create SHAP waterfall plot for a single instance.
        
        Args:
            X (pd.DataFrame or np.ndarray): Data containing the instance
            instance_index (int): Index of instance to explain
            save_path (str): Optional path to save the plot
        """
        if self.shap_values is None:
            self.explain(X)
        
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = self.feature_names
        
        # Create explanation object for waterfall plot
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        explanation = shap.Explanation(
            values=self.shap_values[instance_index],
            base_values=base_value,
            data=X.iloc[instance_index] if isinstance(X, pd.DataFrame) else X[instance_index],
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")
        
        plt.close()
    
    def plot_force(self, X, instance_index=0, save_path=None):
        """
        Create SHAP force plot for a single instance.
        
        Args:
            X (pd.DataFrame or np.ndarray): Data containing the instance
            instance_index (int): Index of instance to explain
            save_path (str): Optional path to save the plot
        """
        if self.shap_values is None:
            self.explain(X)
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        shap.force_plot(
            base_value,
            self.shap_values[instance_index],
            X.iloc[instance_index] if isinstance(X, pd.DataFrame) else X[instance_index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Force plot saved to {save_path}")
        
        plt.close()


class LIMEExplainer:
    """
    LIME-based explainer for customer churn predictions.
    """
    
    def __init__(self, model, feature_names, class_names=['No Churn', 'Churn']):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model with predict_proba method
            feature_names (list): List of feature names
            class_names (list): Names of target classes
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
    
    def create_explainer(self, X_train, mode='classification'):
        """
        Create LIME explainer.
        
        Args:
            X_train (pd.DataFrame or np.ndarray): Training data for LIME
            mode (str): 'classification' or 'regression'
            
        Returns:
            self: Explainer instance
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            random_state=42
        )
        
        return self
    
    def explain_instance(self, instance, num_features=10):
        """
        Explain a single prediction using LIME.
        
        Args:
            instance (pd.Series or np.ndarray): Single instance to explain
            num_features (int): Number of features to include in explanation
            
        Returns:
            Explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        if isinstance(instance, pd.Series):
            instance = instance.values
        elif isinstance(instance, pd.DataFrame):
            instance = instance.values[0]
        
        # Get prediction function
        if hasattr(self.model, 'predict_proba'):
            predict_fn = self.model.predict_proba
        else:
            predict_fn = lambda x: np.column_stack([1 - self.model.predict(x), self.model.predict(x)])
        
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features
        )
        
        return explanation
    
    def plot_explanation(self, explanation, save_path=None):
        """
        Visualize LIME explanation.
        
        Args:
            explanation: LIME explanation object
            save_path (str): Optional path to save the plot
        """
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LIME explanation plot saved to {save_path}")
        
        plt.close()
    
    def get_feature_contributions(self, explanation):
        """
        Get feature contributions from LIME explanation.
        
        Args:
            explanation: LIME explanation object
            
        Returns:
            pd.DataFrame: Feature contributions sorted by absolute impact
        """
        contributions = explanation.as_list()
        
        df = pd.DataFrame(contributions, columns=['feature', 'contribution'])
        df['abs_contribution'] = df['contribution'].abs()
        df = df.sort_values('abs_contribution', ascending=False)
        
        return df[['feature', 'contribution']]


if __name__ == "__main__":
    from model import ChurnPredictor
    from data_processing import generate_sample_churn_data, preprocess_data
    
    # Generate and preprocess data
    print("Generating sample data...")
    data = generate_sample_churn_data(n_samples=1000)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    # Train model
    print("Training model...")
    model = ChurnPredictor(model_type='random_forest')
    model.train(X_train, y_train)
    
    # SHAP Explanations
    print("\nGenerating SHAP explanations...")
    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    shap_explainer.create_explainer(X_train.sample(100), algorithm='tree')
    shap_explainer.explain(X_test[:100])
    shap_explainer.plot_summary(X_test[:100], save_path='visualizations/shap_summary.png')
    shap_explainer.plot_waterfall(X_test, instance_index=0, save_path='visualizations/shap_waterfall.png')
    
    # LIME Explanations
    print("\nGenerating LIME explanations...")
    lime_explainer = LIMEExplainer(model, feature_names=feature_names)
    lime_explainer.create_explainer(X_train)
    explanation = lime_explainer.explain_instance(X_test.iloc[0])
    lime_explainer.plot_explanation(explanation, save_path='visualizations/lime_explanation.png')
    
    print("\nExplainability analysis complete!")
    print(f"Visualizations saved to 'visualizations/' directory")

#!/usr/bin/env python3
"""
Example script demonstrating AI Explainability for Customer Churn Prediction
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import generate_sample_churn_data, preprocess_data
from model import ChurnPredictor
from explainability import SHAPExplainer, LIMEExplainer


def main():
    """
    Main function to demonstrate AI explainability for churn prediction.
    """
    print("="*60)
    print("ChubbChurns - AI Explainability Demo")
    print("="*60)
    
    # Step 1: Generate sample data
    print("\n[1/6] Generating sample customer data...")
    data = generate_sample_churn_data(n_samples=1000, random_state=42)
    print(f"  ✓ Generated {len(data)} customer records")
    print(f"  ✓ Churn rate: {data['churn'].mean():.2%}")
    
    # Step 2: Preprocess data
    print("\n[2/6] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data)
    print(f"  ✓ Training samples: {len(X_train)}")
    print(f"  ✓ Test samples: {len(X_test)}")
    print(f"  ✓ Features: {len(feature_names)}")
    
    # Step 3: Train model
    print("\n[3/6] Training Random Forest model...")
    model = ChurnPredictor(model_type='random_forest', random_state=42)
    model.train(X_train, y_train)
    print("  ✓ Model trained successfully")
    
    # Step 4: Evaluate model
    print("\n[4/6] Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    print("  Model Performance:")
    for metric_name, value in metrics.items():
        print(f"    • {metric_name}: {value:.4f}")
    
    # Step 5: SHAP Explanations
    print("\n[5/6] Generating SHAP explanations...")
    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    shap_explainer.create_explainer(X_train.sample(100, random_state=42), algorithm='tree')
    shap_values = shap_explainer.explain(X_test[:100])
    print("  ✓ SHAP values computed")
    
    # Generate SHAP visualizations
    print("  Generating SHAP visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    shap_explainer.plot_summary(X_test[:100], save_path='visualizations/shap_summary.png')
    shap_explainer.plot_waterfall(X_test, instance_index=0, save_path='visualizations/shap_waterfall_0.png')
    print("  ✓ SHAP plots saved to visualizations/")
    
    # Step 6: LIME Explanations
    print("\n[6/6] Generating LIME explanations...")
    lime_explainer = LIMEExplainer(model, feature_names=feature_names)
    lime_explainer.create_explainer(X_train)
    print("  ✓ LIME explainer created")
    
    # Explain first test instance
    print("\n  Explaining first test instance with LIME...")
    explanation = lime_explainer.explain_instance(X_test.iloc[0], num_features=10)
    lime_explainer.plot_explanation(explanation, save_path='visualizations/lime_explanation_0.png')
    print("  ✓ LIME explanation saved to visualizations/")
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Feature importance
    print("\nTop 5 Most Important Features (from model):")
    feature_importance = model.get_feature_importance()
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Example prediction explanation
    print("\nExample Customer Prediction:")
    instance_idx = 0
    customer_features = X_test.iloc[instance_idx]
    prediction_proba = model.predict_proba(X_test.iloc[[instance_idx]])[0, 1]
    actual_churn = y_test.iloc[instance_idx]
    
    print(f"  Actual Churn: {actual_churn}")
    print(f"  Predicted Probability: {prediction_proba:.4f}")
    print(f"  Prediction: {'CHURN' if prediction_proba >= 0.5 else 'NO CHURN'}")
    
    # LIME contributions
    contributions = lime_explainer.get_feature_contributions(explanation)
    print("\n  Top 3 Contributing Features (LIME):")
    for idx, row in contributions.head(3).iterrows():
        print(f"    • {row['feature']}: {row['contribution']:.4f}")
    
    print("\n" + "="*60)
    print("✓ AI Explainability Demo Complete!")
    print("="*60)
    print("\nGenerated Files:")
    print("  • visualizations/shap_summary.png")
    print("  • visualizations/shap_waterfall_0.png")
    print("  • visualizations/lime_explanation_0.png")
    print("\nNext Steps:")
    print("  • Explore the Jupyter notebook: notebooks/AI_Explainability_Demo.ipynb")
    print("  • Try different models and explainability techniques")
    print("  • Analyze specific customer segments")
    print("="*60)


if __name__ == "__main__":
    main()

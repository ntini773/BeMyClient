"""
ChubbChurns - AI Explainability for Customer Churn Prediction
"""

__version__ = '1.0.0'

from .data_processing import generate_sample_churn_data, preprocess_data
from .model import ChurnPredictor
from .explainability import SHAPExplainer, LIMEExplainer

__all__ = [
    'generate_sample_churn_data',
    'preprocess_data',
    'ChurnPredictor',
    'SHAPExplainer',
    'LIMEExplainer'
]

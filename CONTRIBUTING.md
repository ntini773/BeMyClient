# Contributing to ChubbChurns

Thank you for your interest in contributing to the ChubbChurns AI Explainability project! This hackathon project welcomes contributions from everyone.

## How to Contribute

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ChubbChurns.git
cd ChubbChurns
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
Some ideas for contributions:
- Add new explainability techniques (e.g., Anchors, Counterfactuals)
- Improve model performance
- Add more visualization options
- Enhance documentation
- Create new example notebooks
- Add unit tests
- Optimize performance

### 4. Test Your Changes
```bash
# Make sure your code works
python example.py

# Check that Python files compile
python -m py_compile src/*.py
```

### 5. Commit and Push
```bash
git add .
git commit -m "Add: description of your changes"
git push origin feature/your-feature-name
```

### 6. Create a Pull Request
- Go to your fork on GitHub
- Click "New Pull Request"
- Describe your changes and why they're useful

## Contribution Ideas

### Beginner-Friendly
- [ ] Add more comments to existing code
- [ ] Fix typos in documentation
- [ ] Add more examples to the notebook
- [ ] Create a simple web interface with Streamlit
- [ ] Add data validation checks

### Intermediate
- [ ] Implement new explainability methods (Anchors, Counterfactual Explanations)
- [ ] Add support for different model types (XGBoost, CatBoost, Neural Networks)
- [ ] Create interactive dashboards with Plotly Dash
- [ ] Add model comparison functionality
- [ ] Implement feature selection based on explanations

### Advanced
- [ ] Add support for text/image data beyond tabular
- [ ] Implement custom SHAP kernels for domain-specific models
- [ ] Create real-time explanation API
- [ ] Add adversarial robustness checks
- [ ] Implement explanation quality metrics

## Code Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add type hints where appropriate

Example:
```python
def process_customer_data(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Process customer data for churn prediction.
    
    Args:
        data (pd.DataFrame): Raw customer data
        threshold (float): Classification threshold
        
    Returns:
        pd.DataFrame: Processed data ready for modeling
    """
    # Your code here
    pass
```

## Documentation

- Update README.md if you add major features
- Add docstrings to all new functions
- Update QUICKSTART.md if you change the workflow
- Create examples for new features

## Testing

While we don't have formal unit tests yet, please:
- Test your code manually before submitting
- Ensure example.py runs without errors
- Verify that existing functionality still works
- Document any new dependencies in requirements.txt

## Questions?

Feel free to:
- Open an issue for bugs or feature requests
- Start a discussion for ideas or questions
- Reach out to project maintainers

## Code of Conduct

Be respectful and inclusive:
- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the community

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making ChubbChurns better! 🚀

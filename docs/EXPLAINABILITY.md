# Model Explainability

## SHAP Integration
- Feature importance visualization
- Local explanations for predictions
- Global feature impact analysis

## Usage
```python
explainer = FraudExplainer(model, X_train[:100])
shap_values = explainer.explain_prediction(transaction)
top_features = explainer.get_top_features(shap_values, feature_names)
```

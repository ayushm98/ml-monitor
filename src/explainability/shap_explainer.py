"""SHAP-based model explainability."""
import shap
import numpy as np

class FraudExplainer:
    """Explain fraud predictions using SHAP."""
    
    def __init__(self, model, background_data):
        """Initialize explainer with model and background data."""
        self.model = model
        self.explainer = shap.TreeExplainer(model, background_data)
    
    def explain_prediction(self, features):
        """Generate SHAP values for prediction."""
        shap_values = self.explainer.shap_values(features)
        return shap_values
    
    def get_top_features(self, shap_values, feature_names, top_n=10):
        """Get top contributing features."""
        importance = np.abs(shap_values).mean(0)
        indices = np.argsort(importance)[-top_n:][::-1]
        return [(feature_names[i], importance[i]) for i in indices]

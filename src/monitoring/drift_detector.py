"""Drift detection service for monitoring production data quality."""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitor and detect drift in production data."""

    def __init__(self, reference_data: pd.DataFrame, window_size: int = 1000, psi_threshold: float = 0.1):
        """
        Initialize drift detector.

        Args:
            reference_data: Training/reference dataset
            window_size: Size of sliding window for monitoring
            psi_threshold: PSI threshold for drift detection (0.1 = small shift, 0.2 = medium, 0.25+ = large)
        """
        self.reference_data = reference_data
        self.window_size = window_size
        self.psi_threshold = psi_threshold
        
        # Sliding window for incoming production data
        self.production_window = deque(maxlen=window_size)
        
        # Drift metrics tracking
        self.drift_history = []
        self.alerts = []
        
        # Calculate reference statistics
        self._calculate_reference_stats()
        
        logger.info(f"DriftDetector initialized with window_size={window_size}, threshold={psi_threshold}")

    def _calculate_reference_stats(self):
        """Calculate statistics on reference data."""
        self.reference_stats = {
            'mean': self.reference_data.mean().to_dict(),
            'std': self.reference_data.std().to_dict(),
            'min': self.reference_data.min().to_dict(),
            'max': self.reference_data.max().to_dict(),
            'quantiles': {}
        }
        
        # Calculate quantiles for PSI
        for col in self.reference_data.columns:
            self.reference_stats['quantiles'][col] = np.percentile(
                self.reference_data[col].dropna(), 
                [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            ).tolist()

    def calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        Args:
            expected: Reference distribution
            actual: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI value (0 = no shift, 0.1 = small, 0.2 = medium, 0.25+ = large)
        """
        try:
            # Create bins based on expected distribution
            breakpoints = np.percentile(expected.dropna(), np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)  # Remove duplicates
            
            if len(breakpoints) < 2:
                return 0.0
            
            # Bin both distributions
            expected_counts = pd.cut(expected, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts()
            actual_counts = pd.cut(actual, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts()
            
            # Normalize to proportions
            expected_prop = expected_counts / len(expected)
            actual_prop = actual_counts / len(actual)
            
            # Add small constant to avoid log(0)
            expected_prop = expected_prop + 1e-10
            actual_prop = actual_prop + 1e-10
            
            # Calculate PSI
            psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
            
            return float(psi)
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def add_observation(self, observation: Dict):
        """
        Add a new observation to the monitoring window.

        Args:
            observation: Dictionary with feature values
        """
        self.production_window.append({
            'timestamp': datetime.utcnow().isoformat(),
            'features': observation
        })

    def check_drift(self) -> Dict:
        """
        Check for drift in the current production window.

        Returns:
            Dictionary with drift metrics and alerts
        """
        if len(self.production_window) < 100:  # Need minimum samples
            return {
                'drift_detected': False,
                'message': f'Insufficient data: {len(self.production_window)}/{self.window_size}',
                'features_with_drift': []
            }
        
        # Convert window to DataFrame
        production_df = pd.DataFrame([obs['features'] for obs in self.production_window])
        
        # Calculate PSI for each feature
        drift_scores = {}
        drifted_features = []
        
        for col in self.reference_data.columns:
            if col in production_df.columns:
                psi = self.calculate_psi(self.reference_data[col], production_df[col])
                drift_scores[col] = psi
                
                if psi > self.psi_threshold:
                    drifted_features.append({
                        'feature': col,
                        'psi': round(psi, 4),
                        'severity': self._get_drift_severity(psi)
                    })
        
        # Overall drift status
        max_psi = max(drift_scores.values()) if drift_scores else 0
        drift_detected = max_psi > self.psi_threshold
        
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': drift_detected,
            'max_psi': round(max_psi, 4),
            'window_size': len(self.production_window),
            'features_with_drift': drifted_features,
            'all_psi_scores': {k: round(v, 4) for k, v in drift_scores.items()}
        }
        
        # Log alert if drift detected
        if drift_detected:
            alert_msg = f"DRIFT ALERT: {len(drifted_features)} features drifted (max PSI: {max_psi:.4f})"
            logger.warning(alert_msg)
            self.alerts.append({
                'timestamp': result['timestamp'],
                'message': alert_msg,
                'details': drifted_features
            })
        
        # Track history
        self.drift_history.append(result)
        
        return result

    def _get_drift_severity(self, psi: float) -> str:
        """Categorize drift severity based on PSI."""
        if psi < 0.1:
            return 'none'
        elif psi < 0.2:
            return 'small'
        elif psi < 0.25:
            return 'medium'
        else:
            return 'large'

    def get_drift_report(self) -> Dict:
        """Get comprehensive drift monitoring report."""
        if not self.drift_history:
            return {'status': 'no_data', 'message': 'No drift checks performed yet'}
        
        latest = self.drift_history[-1]
        
        return {
            'latest_check': latest,
            'total_checks': len(self.drift_history),
            'total_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'drift_trend': self._analyze_drift_trend()
        }

    def _analyze_drift_trend(self) -> str:
        """Analyze trend in drift over recent checks."""
        if len(self.drift_history) < 3:
            return 'insufficient_data'
        
        recent_psi = [h['max_psi'] for h in self.drift_history[-10:]]
        
        if len(recent_psi) >= 2:
            trend = np.polyfit(range(len(recent_psi)), recent_psi, 1)[0]
            
            if trend > 0.01:
                return 'increasing'
            elif trend < -0.01:
                return 'decreasing'
            else:
                return 'stable'
        
        return 'unknown'

    def save_state(self, filepath: str):
        """Save detector state to file."""
        state = {
            'drift_history': self.drift_history,
            'alerts': self.alerts,
            'window_size': self.window_size,
            'psi_threshold': self.psi_threshold
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Drift detector state saved to {filepath}")

# Drift Detection and Monitoring

Production monitoring system for detecting data drift and model degradation.

## Overview

The drift detection system continuously monitors incoming production data and compares it against the reference training distribution using Population Stability Index (PSI).

## Components

### DriftDetector

Core drift detection service that:
- Maintains a sliding window of production observations
- Calculates PSI for each feature
- Detects distributional shifts
- Generates alerts when drift exceeds thresholds

### PSI Thresholds

- **< 0.1**: No significant change
- **0.1 - 0.2**: Small shift, monitor closely
- **0.2 - 0.25**: Medium shift, investigate
- **> 0.25**: Large shift, retrain model

## API Endpoints

### GET /drift/status
Returns current drift detection status:
```json
{
  "timestamp": "2025-05-12T10:30:00",
  "drift_detected": false,
  "max_psi": 0.0523,
  "window_size": 1000,
  "features_with_drift": [],
  "all_psi_scores": {
    "Amount": 0.0523,
    "V1": 0.0312,
    ...
  }
}
```

### GET /drift/report
Returns comprehensive drift monitoring report:
```json
{
  "latest_check": {...},
  "total_checks": 15,
  "total_alerts": 2,
  "recent_alerts": [...],
  "drift_trend": "stable"
}
```

## Configuration

```python
drift_detector = DriftDetector(
    reference_data=train_df,
    window_size=1000,      # Number of observations to monitor
    psi_threshold=0.1      # Alert threshold
)
```

## Usage

The drift detector is automatically initialized on API startup and monitors all incoming predictions:

1. Each prediction is added to the monitoring window
2. Periodic drift checks calculate PSI for all features
3. Alerts are logged when drift exceeds threshold
4. State is persisted on shutdown

## Monitoring Best Practices

1. **Check drift status regularly** - Set up automated checks every hour
2. **Investigate alerts promptly** - High PSI indicates model may degrade
3. **Retrain when needed** - Large drift (PSI > 0.25) requires retraining
4. **Track drift trends** - Increasing trend suggests systematic shift
5. **Monitor specific features** - Some features drift more than others

## Alerting

Drift alerts are logged at WARNING level:
```
DRIFT ALERT: 3 features drifted (max PSI: 0.2841)
```

For production, integrate with:
- Email/Slack notifications
- PagerDuty for critical drift
- Dashboard visualizations in Grafana

## State Persistence

Drift detector state is saved to `data/drift_state.json` on shutdown, including:
- Drift history
- All alerts
- Configuration

## Future Enhancements

- Prediction drift monitoring (output distribution)
- Label drift detection (when ground truth available)
- Automated retraining triggers
- Feature-specific alert thresholds
- Drift visualization dashboard

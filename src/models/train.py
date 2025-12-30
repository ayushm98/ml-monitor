"""Train fraud detection models with MLflow tracking."""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import FraudDataLoader, FraudFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """Train and evaluate fraud detection models."""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        """
        Initialize trainer.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mlflow_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.experiment_name = "fraud-detection"

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not connect to MLflow: {e}")
            self.experiment_id = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data for training.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Loading fraud detection dataset...")

        # Load raw data
        loader = FraudDataLoader("data/raw")
        df = loader.load_raw_data()

        # Get data info
        info = loader.get_data_info(df)
        logger.info(f"Dataset: {info['num_transactions']:,} transactions, "
                   f"{info['fraud_percentage']:.4f}% fraud rate")

        # Split data
        train_df, test_df = loader.create_train_test_split(df, test_size=0.2)

        # Feature engineering
        engineer = FraudFeatureEngineer()

        # Prepare train set
        train_prepared, feature_names = engineer.prepare_features(
            train_df,
            include_engineered=True,
            scale=True,
            fit_scaler=True
        )
        X_train, y_train = engineer.get_X_y(train_prepared)

        # Prepare test set (use fitted scaler)
        test_prepared, _ = engineer.prepare_features(
            test_df,
            include_engineered=True,
            scale=True,
            fit_scaler=False
        )
        X_test, y_test = engineer.get_X_y(test_prepared)

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Features: {len(feature_names)}")

        return X_train, X_test, y_train, y_test

    def train_baseline_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 10,
        use_smote: bool = True
    ) -> ImbPipeline:
        """
        Train baseline RandomForest with SMOTE.

        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Max tree depth
            use_smote: Whether to use SMOTE for balancing

        Returns:
            Trained pipeline
        """
        logger.info("Training baseline RandomForest model...")

        # Create pipeline with SMOTE + RandomForest
        steps = []

        if use_smote:
            # SMOTE to oversample minority class to 0.1 ratio (instead of 1:1)
            # This balances between handling imbalance and not overwhelming majority class
            steps.append(('smote', SMOTE(random_state=42, sampling_strategy=0.1)))

        # RandomForest classifier
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        steps.append(('classifier', rf))

        # Create pipeline
        pipeline = ImbPipeline(steps)

        # Train
        pipeline.fit(X_train, y_train)

        logger.info("Model training complete")
        return pipeline

    def evaluate_model(
        self,
        model: ImbPipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dict of metrics
        """
        logger.info("Evaluating model...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        })

        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def train_with_mlflow(
        self,
        run_name: str = "baseline-rf",
        n_estimators: int = 100,
        max_depth: int = 10,
        use_smote: bool = True
    ):
        """
        Full training pipeline with MLflow tracking.

        Args:
            run_name: MLflow run name
            n_estimators: Number of trees
            max_depth: Max tree depth
            use_smote: Whether to use SMOTE
        """
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()

        # Start MLflow run
        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id):
            # Log parameters
            params = {
                'model_type': 'RandomForest',
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'use_smote': use_smote,
                'class_weight': 'balanced',
                'n_features': X_train.shape[1],
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'fraud_rate_train': y_train.mean(),
                'fraud_rate_test': y_test.mean(),
            }
            mlflow.log_params(params)

            # Train model
            model = self.train_baseline_model(
                X_train, y_train,
                n_estimators=n_estimators,
                max_depth=max_depth,
                use_smote=use_smote
            )

            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="fraud-detector"
            )

            logger.info(f"MLflow run completed: {mlflow.active_run().info.run_id}")
            logger.info(f"Model registered as: fraud-detector")

            return model, metrics


def main():
    """Run training pipeline."""
    print("\n" + "=" * 60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 60 + "\n")

    # Initialize trainer
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    trainer = FraudModelTrainer(mlflow_tracking_uri=mlflow_uri)

    # Train baseline model
    print("Training baseline RandomForest with SMOTE...\n")
    model, metrics = trainer.train_with_mlflow(
        run_name="baseline-rf-smote",
        n_estimators=100,
        max_depth=10,
        use_smote=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print("=" * 60 + "\n")

    print("View results: http://localhost:5000")


if __name__ == "__main__":
    main()

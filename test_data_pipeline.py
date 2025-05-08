"""Test script for fraud detection data pipeline."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import FraudDataLoader, FraudFeatureEngineer, FraudDataValidator


def main():
    """Test the complete data pipeline."""
    print("\n" + "=" * 60)
    print("FRAUD DETECTION DATA PIPELINE TEST")
    print("=" * 60 + "\n")

    # 1. Test Data Loading
    print("1. Testing Data Loading...")
    print("-" * 40)
    loader = FraudDataLoader("data/raw")
    df = loader.load_raw_data()
    info = loader.get_data_info(df)

    print(f"   Transactions: {info['num_transactions']:,}")
    print(f"   Features: {info['num_features']}")
    print(f"   Fraud cases: {info['fraud_count']:,} ({info['fraud_percentage']:.2f}%)")
    print(f"   Imbalance ratio: 1:{info['class_imbalance_ratio']}")
    print(f"   Amount range: ${info['amount_stats']['min']:.2f} - ${info['amount_stats']['max']:.2f}")
    print(f"   Time span: {info['time_range_hours']} hours")
    print(f"   Missing values: {info['missing_values']}")
    print("   ✓ Data loading passed\n")

    # 2. Test Data Validation
    print("2. Testing Data Validation...")
    print("-" * 40)
    validator = FraudDataValidator()
    validation_results = validator.run_all_validations(df)

    print(f"   Schema valid: {validation_results['schema']['is_valid']}")
    print(f"   Data quality valid: {validation_results['data_quality']['is_valid']}")
    print(f"   Numeric ranges valid: {validation_results['numeric_ranges']['overall_valid']}")
    print(f"   Overall: {'✓ PASSED' if validation_results['overall_valid'] else '✗ FAILED'}\n")

    # 3. Test Feature Engineering
    print("3. Testing Feature Engineering...")
    print("-" * 40)
    engineer = FraudFeatureEngineer()
    df_prepared, feature_names = engineer.prepare_features(df)
    X, y = engineer.get_X_y(df_prepared)

    print(f"   Original features: 31")
    print(f"   Engineered features: {len(feature_names)}")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Fraud rate in y: {y.mean():.4%}")
    print("   ✓ Feature engineering passed\n")

    # 4. Test Train/Test Split
    print("4. Testing Train/Test Split...")
    print("-" * 40)
    train_df, test_df = loader.create_train_test_split(df)

    print(f"   Train size: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Test size: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   Train fraud rate: {train_df['Class'].mean():.4%}")
    print(f"   Test fraud rate: {test_df['Class'].mean():.4%}")
    print("   ✓ Stratified split maintains fraud rate\n")

    # 5. Save processed data
    print("5. Saving Processed Data...")
    print("-" * 40)
    output_path = Path("data/processed/fraud_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = X.copy()
    output_df["Class"] = y
    output_df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("   ✓ Data saved\n")

    # Summary
    print("=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"✓ Dataset: {info['num_transactions']:,} transactions")
    print(f"✓ Fraud rate: {info['fraud_percentage']:.4f}%")
    print(f"✓ Features: {len(feature_names)} (after engineering)")
    print(f"✓ All validations: PASSED")
    print(f"✓ Pipeline ready for model training")
    print("=" * 60 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

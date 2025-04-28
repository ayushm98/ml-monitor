"""Test script for data pipeline."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import BikeShareDataLoader, BikeShareFeatureEngineer, DataValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("="*60)
    print("Testing ML-Monitor Data Pipeline")
    print("="*60)

    # 1. Load data
    print("\n1. Loading raw data...")
    loader = BikeShareDataLoader(data_path="data/raw")
    df = loader.load_raw_data()
    info = loader.get_data_info(df)
    print(f"   Loaded {info['num_records']:,} records")
    print(f"   Date range: {info['date_range']['start']} to {info['date_range']['end']}")
    print(f"   Memory usage: {info['memory_usage_mb']:.2f} MB")

    # 2. Validate data
    print("\n2. Validating data quality...")
    validator = DataValidator()
    validation_results = validator.run_all_validations(df)
    print(f"   Schema valid: {validation_results['schema']['is_valid']}")
    print(f"   Timestamp valid: {validation_results['timestamp_validation']['is_valid']}")
    print(f"   Overall validation: {'PASSED ✓' if validation_results['overall_valid'] else 'FAILED ✗'}")

    # 3. Engineer features
    print("\n3. Engineering features...")
    engineer = BikeShareFeatureEngineer()
    df_features = engineer.create_aggregated_features(df)
    print(f"   Created {len(df_features)} hourly records")
    print(f"   Features shape: {df_features.shape}")

    # 4. Prepare ML dataset
    print("\n4. Preparing ML dataset...")
    X, y, feature_names = engineer.prepare_training_data(df_features)
    print(f"   Training samples: {len(X):,}")
    print(f"   Number of features: {len(feature_names)}")
    print(f"   Feature names: {feature_names[:5]}...")  # Show first 5
    print(f"   Target statistics:")
    if y is not None:
        print(f"     Mean: {y.mean():.2f}")
        print(f"     Std: {y.std():.2f}")
        print(f"     Min: {y.min():.2f}")
        print(f"     Max: {y.max():.2f}")

    # 5. Save processed data
    print("\n5. Saving processed data...")
    output_path = Path("data/processed/bikeshare_processed.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ml_data = X.copy()
    if y is not None:
        ml_data['target'] = y
    ml_data.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n" + "="*60)
    print("Data pipeline test completed successfully! ✓")
    print("="*60)

if __name__ == "__main__":
    main()

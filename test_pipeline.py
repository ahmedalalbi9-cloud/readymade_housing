"""
Test script for California Housing Analysis Pipeline
This script validates that all components are working correctly
"""

import sys
import traceback

def test_imports():
    """Test if all required libraries can be imported"""
    print("\n" + "="*80)
    print("TEST 1: Testing imports...")
    print("="*80)
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
        
        import seaborn as sns
        print("✓ Seaborn imported successfully")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        print("✓ Scikit-learn components imported successfully")
        
        print("\n✓ All imports successful!")
        return True
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "="*80)
    print("TEST 2: Testing data loading...")
    print("="*80)
    
    try:
        from housing_analysis import load_data
        df = load_data()
        
        assert df is not None, "DataFrame is None"
        assert df.shape[0] > 0, "DataFrame has no rows"
        assert df.shape[1] >= 9, "DataFrame has insufficient columns"
        assert 'MedHouseVal' in df.columns, "Target column missing"
        
        print(f"\n✓ Data loaded successfully: {df.shape}")
        return True
    except Exception as e:
        print(f"\n✗ Data loading failed: {e}")
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n" + "="*80)
    print("TEST 3: Testing feature engineering...")
    print("="*80)
    
    try:
        from housing_analysis import load_data, engineer_features
        df = load_data()
        df_engineered = engineer_features(df)
        
        expected_features = ['RoomsPerHousehold', 'BedroomsToRooms', 
                           'PopulationPerHousehold', 'TotalRoomsIndicator',
                           'IncomeAgeInteraction', 'GeoDensity']
        
        for feature in expected_features:
            assert feature in df_engineered.columns, f"Feature {feature} not found"
            print(f"✓ Feature '{feature}' created successfully")
        
        assert df_engineered.shape[1] == df.shape[1] + 6, "Incorrect number of features"
        
        print(f"\n✓ All features engineered successfully!")
        print(f"  Original features: {df.shape[1]}")
        print(f"  New features: {df_engineered.shape[1]}")
        return True
    except Exception as e:
        print(f"\n✗ Feature engineering failed: {e}")
        traceback.print_exc()
        return False


def test_model_training():
    """Test model training functionality"""
    print("\n" + "="*80)
    print("TEST 4: Testing model training...")
    print("="*80)
    
    try:
        from housing_analysis import load_data, engineer_features, train_models
        
        df = load_data()
        df_engineered = engineer_features(df)
        lr_model, rf_model, scaler = train_models(df_engineered)
        
        assert lr_model is not None, "Linear Regression model is None"
        assert rf_model is not None, "Random Forest model is None"
        assert scaler is not None, "Scaler is None"
        
        print("\n✓ All models trained successfully!")
        print(f"  • Linear Regression: {type(lr_model).__name__}")
        print(f"  • Random Forest: {type(rf_model).__name__}")
        print(f"  • Scaler: {type(scaler).__name__}")
        return True
    except Exception as e:
        print(f"\n✗ Model training failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("CALIFORNIA HOUSING ANALYSIS - TEST SUITE")
    print("="*80)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Feature Engineering Test", test_feature_engineering),
        ("Model Training Test", test_model_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*80)
    print(f"Results: {passed}/{total} tests passed ({passed*100//total}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

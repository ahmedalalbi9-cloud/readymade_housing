"""
California Housing Price Prediction
Data exploration, cleaning, processing, feature engineering, and machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def load_data():
    """Load the California housing dataset"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    try:
        # Try to fetch from sklearn if online
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        print(f"\nDataset loaded successfully from sklearn!")
    except Exception as e:
        # If offline or fetch fails, create sample data
        print(f"\nNote: Could not fetch data online. Creating sample dataset...")
        np.random.seed(42)
        n_samples = 20640
        
        # Generate sample data with realistic distributions
        df = pd.DataFrame({
            'MedInc': np.random.exponential(scale=3.8, size=n_samples),
            'HouseAge': np.random.uniform(1, 52, size=n_samples),
            'AveRooms': np.random.normal(5.4, 2.5, size=n_samples).clip(1, 20),
            'AveBedrms': np.random.normal(1.1, 0.5, size=n_samples).clip(0.5, 5),
            'Population': np.random.exponential(scale=1400, size=n_samples),
            'AveOccup': np.random.normal(3.0, 1.5, size=n_samples).clip(1, 10),
            'Latitude': np.random.uniform(32.5, 42.0, size=n_samples),
            'Longitude': np.random.uniform(-124.3, -114.3, size=n_samples),
        })
        
        # Generate target variable with realistic relationships
        df['MedHouseVal'] = (
            2.5 * df['MedInc'] + 
            0.01 * df['HouseAge'] - 
            0.05 * df['Population'] / 1000 +
            0.2 * df['AveRooms'] +
            np.random.normal(0, 0.5, size=n_samples)
        ).clip(0.15, 5.0)
        
        print(f"\nSample dataset created successfully!")
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target: MedHouseVal (Median House Value)")
    
    return df


def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*80)
    print("DATA EXPLORATION")
    print("="*80)
    
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- Dataset Info ---")
    print(df.info())
    
    print("\n--- Statistical Summary ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Correlation with Target (MedHouseVal) ---")
    correlation = df.corr()['MedHouseVal'].sort_values(ascending=False)
    print(correlation)
    
    return df


def clean_data(df):
    """Clean and preprocess the data"""
    print("\n" + "="*80)
    print("DATA CLEANING AND PROCESSING")
    print("="*80)
    
    print("\n--- Checking for duplicates ---")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    print("\n--- Checking for outliers using IQR method ---")
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    outliers_count = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print("Outliers per column:")
    print(outliers_count)
    
    print("\n--- Data types ---")
    print(df.dtypes)
    
    print("\nData cleaning completed!")
    return df


def engineer_features(df):
    """Create new features by combining existing columns"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING - CREATING NEW COMBINED FEATURES")
    print("="*80)
    
    df_new = df.copy()
    
    # Feature 1: Rooms per household
    df_new['RoomsPerHousehold'] = df_new['AveRooms'] * df_new['AveOccup']
    print("\n✓ Created 'RoomsPerHousehold' = AveRooms * AveOccup")
    
    # Feature 2: Bedrooms to Rooms ratio
    df_new['BedroomsToRooms'] = df_new['AveBedrms'] / (df_new['AveRooms'] + 1e-10)
    print("✓ Created 'BedroomsToRooms' = AveBedrms / AveRooms")
    
    # Feature 3: Population per household
    df_new['PopulationPerHousehold'] = df_new['Population'] / (df_new['AveOccup'] + 1e-10)
    print("✓ Created 'PopulationPerHousehold' = Population / AveOccup")
    
    # Feature 4: Total rooms (combining average rooms with occupancy indicator)
    df_new['TotalRoomsIndicator'] = df_new['AveRooms'] + df_new['AveBedrms']
    print("✓ Created 'TotalRoomsIndicator' = AveRooms + AveBedrms")
    
    # Feature 5: Income-Age interaction
    df_new['IncomeAgeInteraction'] = df_new['MedInc'] * df_new['HouseAge']
    print("✓ Created 'IncomeAgeInteraction' = MedInc * HouseAge")
    
    # Feature 6: Geographic density
    df_new['GeoDensity'] = df_new['Population'] / (df_new['Latitude'] * df_new['Longitude'] + 1e-10)
    print("✓ Created 'GeoDensity' = Population / (Latitude * Longitude)")
    
    print(f"\n--- New dataset shape: {df_new.shape} ---")
    print(f"Added {df_new.shape[1] - df.shape[1]} new features")
    
    print("\n--- New features correlation with target ---")
    new_features = ['RoomsPerHousehold', 'BedroomsToRooms', 'PopulationPerHousehold', 
                    'TotalRoomsIndicator', 'IncomeAgeInteraction', 'GeoDensity']
    for feature in new_features:
        corr = df_new[feature].corr(df_new['MedHouseVal'])
        print(f"{feature}: {corr:.4f}")
    
    return df_new


def train_models(df):
    """Train machine learning models"""
    print("\n" + "="*80)
    print("MACHINE LEARNING - MODEL TRAINING")
    print("="*80)
    
    # Prepare data
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Linear Regression
    print("\n" + "-"*80)
    print("Training Linear Regression Model...")
    print("-"*80)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    lr_train_pred = lr_model.predict(X_train_scaled)
    lr_test_pred = lr_model.predict(X_test_scaled)
    
    lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
    lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
    lr_train_r2 = r2_score(y_train, lr_train_pred)
    lr_test_r2 = r2_score(y_test, lr_test_pred)
    lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
    
    print(f"\nLinear Regression Results:")
    print(f"  Training RMSE: {lr_train_rmse:.4f}")
    print(f"  Test RMSE: {lr_test_rmse:.4f}")
    print(f"  Training R²: {lr_train_r2:.4f}")
    print(f"  Test R²: {lr_test_r2:.4f}")
    print(f"  Test MAE: {lr_test_mae:.4f}")
    
    # Model 2: Random Forest Regressor
    print("\n" + "-"*80)
    print("Training Random Forest Regressor Model...")
    print("-"*80)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train_scaled, y_train)
    
    rf_train_pred = rf_model.predict(X_train_scaled)
    rf_test_pred = rf_model.predict(X_test_scaled)
    
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
    
    print(f"\nRandom Forest Results:")
    print(f"  Training RMSE: {rf_train_rmse:.4f}")
    print(f"  Test RMSE: {rf_test_rmse:.4f}")
    print(f"  Training R²: {rf_train_r2:.4f}")
    print(f"  Test R²: {rf_test_r2:.4f}")
    print(f"  Test MAE: {rf_test_mae:.4f}")
    
    # Feature importance
    print("\n--- Top 10 Feature Importances (Random Forest) ---")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Model comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<25} {'Test RMSE':<15} {'Test R²':<15} {'Test MAE':<15}")
    print("-"*80)
    print(f"{'Linear Regression':<25} {lr_test_rmse:<15.4f} {lr_test_r2:<15.4f} {lr_test_mae:<15.4f}")
    print(f"{'Random Forest':<25} {rf_test_rmse:<15.4f} {rf_test_r2:<15.4f} {rf_test_mae:<15.4f}")
    
    if rf_test_rmse < lr_test_rmse:
        print(f"\n✓ Random Forest performs better with lower RMSE!")
    else:
        print(f"\n✓ Linear Regression performs better with lower RMSE!")
    
    return lr_model, rf_model, scaler


def main():
    """Main function to run the complete pipeline"""
    print("\n" + "="*80)
    print("CALIFORNIA HOUSING PRICE PREDICTION")
    print("Complete Data Science Pipeline")
    print("="*80)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Explore data
    df = explore_data(df)
    
    # Step 3: Clean data
    df = clean_data(df)
    
    # Step 4: Engineer new features
    df_engineered = engineer_features(df)
    
    # Step 5: Train machine learning models
    lr_model, rf_model, scaler = train_models(df_engineered)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nSummary:")
    print(f"  • Loaded {df.shape[0]} samples with {df.shape[1]} features")
    print(f"  • Created {df_engineered.shape[1] - df.shape[1]} new engineered features")
    print(f"  • Trained 2 machine learning models (Linear Regression, Random Forest)")
    print(f"  • Final dataset shape: {df_engineered.shape}")
    
    return df, df_engineered, lr_model, rf_model, scaler


if __name__ == "__main__":
    df, df_engineered, lr_model, rf_model, scaler = main()

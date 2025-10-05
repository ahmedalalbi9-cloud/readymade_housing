# California Housing Price Prediction

A comprehensive data science project for predicting median housing prices in California. This project demonstrates a complete machine learning pipeline including data exploration, cleaning, processing, feature engineering, and model training.

## Project Overview

This project analyzes the California Housing dataset and builds predictive models to estimate median house values. The pipeline includes:

- **Data Exploration**: Comprehensive analysis of the dataset including distributions, correlations, and statistical summaries
- **Data Cleaning**: Detection and handling of duplicates, missing values, and outliers
- **Feature Engineering**: Creation of 6 new features by combining existing columns
- **Machine Learning**: Training and evaluation of multiple regression models

## Features

### Original Features
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

### Engineered Features (New Combinations)
1. **RoomsPerHousehold**: `AveRooms * AveOccup` - Total rooms accounting for occupancy
2. **BedroomsToRooms**: `AveBedrms / AveRooms` - Bedroom to room ratio
3. **PopulationPerHousehold**: `Population / AveOccup` - Household density
4. **TotalRoomsIndicator**: `AveRooms + AveBedrms` - Combined room metric
5. **IncomeAgeInteraction**: `MedInc * HouseAge` - Income-age interaction term
6. **GeoDensity**: `Population / (Latitude * Longitude)` - Geographic density measure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ahmedalalbi9-cloud/readymade_housing.git
cd readymade_housing
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Python Script

Execute the complete pipeline:
```bash
python housing_analysis.py
```

This will:
- Load the California Housing dataset
- Perform exploratory data analysis
- Clean and process the data
- Engineer new features
- Train and evaluate machine learning models
- Display comprehensive results and metrics

### Using the Jupyter Notebook

For interactive analysis:
```bash
jupyter notebook housing_analysis.ipynb
```

The notebook includes:
- Step-by-step analysis with visualizations
- Interactive exploration of features
- Detailed model comparisons
- Feature importance analysis

## Project Structure

```
readymade_housing/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
├── housing_analysis.py         # Main Python script
└── housing_analysis.ipynb      # Jupyter notebook
```

## Machine Learning Models

The project implements two regression models:

1. **Linear Regression**: Baseline model for comparison
2. **Random Forest Regressor**: Ensemble model with better performance

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Measures prediction accuracy
- **R² Score**: Coefficient of determination (model fit quality)
- **MAE** (Mean Absolute Error): Average prediction error

## Results

The pipeline processes 20,640+ housing samples with 14 total features (8 original + 6 engineered). The Random Forest model typically achieves better performance than Linear Regression, benefiting from the engineered features.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter (for notebook)

See `requirements.txt` for specific versions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Ahmed Alalbi

## Acknowledgments

- Dataset: California Housing Prices from scikit-learn
- Inspired by real-world housing price prediction challenges

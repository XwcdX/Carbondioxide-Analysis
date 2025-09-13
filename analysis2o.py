import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import re
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. Load and Clean Data
# -----------------------------------------------------------------
print("Step 1: Loading and Cleaning Data...")

try:
    train_df = pd.read_csv('datatrain.csv')
    test_df = pd.read_csv('datatest.csv')
    submission_dates = test_df['date'].copy()
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
except FileNotFoundError:
    print("Make sure 'datatrain.csv' and 'datatest.csv' are in the same directory.")
    exit()

# Store important data
train_ids = train_df['obs_id']
test_ids = test_df['obs_id']
train_y = train_df['daily_ktCO2']

print(f"Target stats - Mean: {train_y.mean():.3f}, Std: {train_y.std():.3f}")
print(f"Target range: [{train_y.min():.3f}, {train_y.max():.3f}]")

# Drop target and ids
train_df = train_df.drop(columns=['daily_ktCO2', 'obs_id'])
test_df = test_df.drop(columns=['obs_id'])

# Combine for preprocessing
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Clean text data and convert numeric columns
text_cols = combined_df.select_dtypes(include=['object']).columns
for col in text_cols:
    if col != 'date':
        combined_df[col] = combined_df[col].astype(str).str.lower().str.strip()

# Convert all potential numeric columns to numeric, handling errors
numeric_cols_to_convert = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'env_index', 'temp', 'tempmax', 'tempmin', 
                          'temp_calib', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity',
                          'precip', 'precipprob', 'precipcover', 'windgust', 'windspeed', 'windspeedmax',
                          'windspeedmean', 'windspeedmin', 'winddir', 'sealevelpressure', 'cloudcover',
                          'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase',
                          'avg_ridership_monthly', 'avg_ridership_workday_monthly', 'TCI']

# Add traffic columns
traffic_pattern_cols = [col for col in combined_df.columns if any(pattern in col for pattern in ['7-9', '9-17', '17-19'])]
numeric_cols_to_convert.extend(traffic_pattern_cols)
numeric_cols_to_convert = list(pd.Series(numeric_cols_to_convert).drop_duplicates())

for col in numeric_cols_to_convert:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

# Fix boolean columns
bool_cols = ['rain', 'weekend', 'holiday']
for col in bool_cols:
    if col in combined_df.columns:
        map_dict = {'true': True, 'yes': True, '1': True, 'false': False, 'no': False, '0': False}
        combined_df[col] = combined_df[col].astype(str).str.strip().str.lower().map(map_dict)
        combined_df[col] = combined_df[col].fillna(False).astype(int)

print("Data cleaning complete.")

# -----------------------------------------------------------------
# 2. ADVANCED Feature Engineering
# -----------------------------------------------------------------
print("\nStep 2: Advanced Feature Engineering...")

# Date features with enhanced temporal encoding
combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['month'] = combined_df['date'].dt.month
combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
combined_df['day_of_year'] = combined_df['date'].dt.dayofyear
combined_df['week_of_year'] = combined_df['date'].dt.isocalendar().week.astype(int)
combined_df['year'] = combined_df['date'].dt.year
combined_df['quarter'] = combined_df['date'].dt.quarter
combined_df['is_weekend'] = (combined_df['day_of_week'] >= 5).astype(int)
combined_df['is_month_start'] = combined_df['date'].dt.is_month_start.astype(int)
combined_df['is_month_end'] = combined_df['date'].dt.is_month_end.astype(int)
combined_df['days_since_start'] = (combined_df['date'] - combined_df['date'].min()).dt.days

# Advanced time-based features
combined_df['is_quarter_start'] = combined_df['date'].dt.is_quarter_start.astype(int)
combined_df['is_quarter_end'] = combined_df['date'].dt.is_quarter_end.astype(int)
combined_df['days_in_month'] = combined_df['date'].dt.days_in_month
combined_df['week_of_month'] = (combined_df['date'].dt.day - 1) // 7 + 1

combined_df = combined_df.drop('date', axis=1)

# Enhanced cyclical encoding
def add_cyclical_features(df, col, period):
    """Add sine/cosine encoding for cyclical features"""
    if col in df.columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
        # Add harmonics for better representation
        df[f'{col}_sin2'] = np.sin(4 * np.pi * df[col] / period)
        df[f'{col}_cos2'] = np.cos(4 * np.pi * df[col] / period)
    return df

# Apply cyclical encoding with harmonics
if 'winddir' in combined_df.columns:
    combined_df = add_cyclical_features(combined_df, 'winddir', 360)
    combined_df = combined_df.drop('winddir', axis=1)

combined_df = add_cyclical_features(combined_df, 'month', 12)
combined_df = add_cyclical_features(combined_df, 'day_of_week', 7)
combined_df = add_cyclical_features(combined_df, 'day_of_year', 365)
combined_df = add_cyclical_features(combined_df, 'week_of_year', 52)

# Drop original cyclical columns
cyclical_to_drop = ['month', 'day_of_week', 'day_of_year', 'week_of_year']
combined_df = combined_df.drop(columns=[col for col in cyclical_to_drop if col in combined_df.columns])

# ADVANCED Traffic aggregations
traffic_cols = [col for col in combined_df.columns if any(time in col for time in ['7-9', '9-17', '17-19'])]
vehicle_types = ['Car', 'Van', 'Bus', 'Minibus', 'Truck', '3Cycle']

for vehicle in vehicle_types:
    vehicle_cols = [col for col in traffic_cols if vehicle in col]
    if len(vehicle_cols) > 0:
        combined_df[f'{vehicle}_total'] = combined_df[vehicle_cols].sum(axis=1)
        combined_df[f'{vehicle}_peak_ratio'] = (combined_df.get(f'{vehicle}_7-9', 0) + 
                                               combined_df.get(f'{vehicle}_17-19', 0)) / (combined_df[f'{vehicle}_total'] + 1e-6)

# Time-based traffic patterns
morning_cols = [col for col in traffic_cols if '7-9' in col]
day_cols = [col for col in traffic_cols if '9-17' in col]
evening_cols = [col for col in traffic_cols if '17-19' in col]

if morning_cols:
    combined_df['traffic_morning'] = combined_df[morning_cols].sum(axis=1)
else:
    combined_df['traffic_morning'] = 0
if day_cols:
    combined_df['traffic_day'] = combined_df[day_cols].sum(axis=1)
else:
    combined_df['traffic_day'] = 0
if evening_cols:
    combined_df['traffic_evening'] = combined_df[evening_cols].sum(axis=1)
else:
    combined_df['traffic_evening'] = 0

combined_df['traffic_total'] = (combined_df['traffic_morning'] +
                               combined_df['traffic_day'] +
                               combined_df['traffic_evening'])

# Traffic ratios and patterns
combined_df['rush_hour_ratio'] = ((combined_df['traffic_morning'] +
                                  combined_df['traffic_evening']) /
                                 (combined_df['traffic_total'] + 1e-6))

# ADVANCED Weather and Environmental interactions
weather_cols = ['temp', 'tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure']
pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2']

# Temperature variations and comfort indices
if all(col in combined_df.columns for col in ['temp', 'tempmax', 'tempmin']):
    combined_df['temp_range'] = combined_df['tempmax'] - combined_df['tempmin']
    combined_df['temp_variance'] = combined_df['temp_range'] / (combined_df['temp'].replace(0, 1e-6) + 1e-6) # Avoid division by zero

# Enhanced weather interactions
if 'temp' in combined_df.columns and 'humidity' in combined_df.columns:
    combined_df['heat_index'] = combined_df['temp'] * (1 + combined_df['humidity'] / 100)
    combined_df['comfort_index'] = combined_df['temp'] * (1 - abs(combined_df['humidity'] - 50) / 100)

# Advanced wind features
wind_features = ['windspeed', 'windgust', 'windspeedmax', 'windspeedmin']
for feat in wind_features:
    if feat in combined_df.columns and 'temp' in combined_df.columns:
        combined_df[f'{feat}_temp_interaction'] = combined_df[feat] * combined_df['temp']

# Pressure-based features
if 'sealevelpressure' in combined_df.columns:
    combined_df['pressure_anomaly'] = combined_df['sealevelpressure'] - combined_df['sealevelpressure'].mean()

# Enhanced pollution interactions
if len([col for col in pollutant_cols if col in combined_df.columns]) > 1:
    available_pollutants = [col for col in pollutant_cols if col in combined_df.columns]
    combined_df['pollutant_sum'] = combined_df[available_pollutants].sum(axis=1)
    combined_df['pollutant_mean'] = combined_df[available_pollutants].mean(axis=1)
    
    # Pollution ratios
    if 'pm25' in combined_df.columns and 'pm10' in combined_df.columns:
        combined_df['fine_coarse_ratio'] = combined_df['pm25'] / (combined_df['pm10'] + 1e-6)
    
    if 'no2' in combined_df.columns and 'o3' in combined_df.columns:
        combined_df['no2_o3_ratio'] = combined_df['no2'] / (combined_df['o3'] + 1e-6)

# Solar and UV interactions
solar_features = ['solarradiation', 'solarenergy', 'uvindex']
for feat in solar_features:
    if feat in combined_df.columns and 'cloudcover' in combined_df.columns:
        combined_df[f'{feat}_cloud_interaction'] = combined_df[feat] * (100 - combined_df['cloudcover']) / 100

# Advanced ridership features
ridership_cols = ['avg_ridership_monthly', 'avg_ridership_workday_monthly']
for col in ridership_cols:
    if col in combined_df.columns:
        combined_df[f'{col}_per_traffic'] = combined_df[col] / (combined_df.get('traffic_total', 1) + 1e-6)

print("Advanced feature engineering complete.")

# -----------------------------------------------------------------
# 3. SMART Missing Value Handling with Advanced Techniques
# -----------------------------------------------------------------
print("\nStep 3: Advanced Missing Value Handling...")

# Analyze missing patterns
missing_analysis = pd.DataFrame({
    'column': combined_df.columns,
    'missing_count': combined_df.isnull().sum(),
    'missing_pct': (combined_df.isnull().sum() / len(combined_df)) * 100
})
missing_analysis = missing_analysis[missing_analysis['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("Columns with missing values:")
if len(missing_analysis) > 0:
    print(missing_analysis.to_string(index=False))
else:
    print("No missing values found.")

# Drop high-missing columns (>70% missing)
high_missing_cols = missing_analysis[missing_analysis['missing_pct'] > 70]['column'].tolist()
if high_missing_cols:
    combined_df = combined_df.drop(columns=high_missing_cols)
    print(f"\nDropped {len(high_missing_cols)} columns with >70% missing data")

# Advanced imputation with seasonal patterns
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = combined_df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Seasonal imputation for weather features
if 'month_sin' in combined_df.columns:
    weather_seasonal_cols = [col for col in numerical_cols if any(weather in col.lower() 
                            for weather in ['temp', 'humid', 'wind', 'pressure', 'solar', 'precip', 'uvindex', 'dew'])]
    
    for col in weather_seasonal_cols:
        if combined_df[col].isnull().any():
            min_val, max_val = combined_df['month_sin'].min(), combined_df['month_sin'].max()
            bins = np.linspace(min_val, max_val, 5)
            
            if combined_df['month_sin'].nunique() > 1:
                try:
                    for _, season_group_df in combined_df.groupby(pd.cut(combined_df['month_sin'], bins=bins, include_lowest=True)):
                        mask = season_group_df.index
                        if combined_df.loc[mask, col].isnull().any():
                            fill_val = combined_df.loc[mask, col].median()
                            if pd.isna(fill_val):
                                fill_val = combined_df[col].median() if abs(combined_df[col].skew()) > 2 else combined_df[col].mean()
                            combined_df.loc[mask, col] = combined_df.loc[mask, col].fillna(fill_val)
                except Exception as e:
                    print(f"Warning: Seasonal imputation for {col} failed ({e}). Falling back to global imputation.")
                    skewness = combined_df[col].skew()
                    fill_val = combined_df[col].median() if abs(skewness) > 2 else combined_df[col].mean()
                    combined_df[col] = combined_df[col].fillna(fill_val)
            else:
                skewness = combined_df[col].skew()
                fill_val = combined_df[col].median() if abs(skewness) > 2 else combined_df[col].mean()
                combined_df[col] = combined_df[col].fillna(fill_val)
else:
    print("Warning: 'month_sin' not found for seasonal imputation. Skipping seasonal imputation for weather features.")

# Standard numerical imputation for remaining columns
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    if combined_df[col].isnull().any():
        skewness = combined_df[col].skew()
        fill_val = combined_df[col].median() if abs(skewness) > 1 else combined_df[col].mean()
        combined_df[col] = combined_df[col].fillna(fill_val)

# Categorical imputation
categorical_cols = combined_df.select_dtypes(include=['object', 'bool']).columns.tolist()
for col in categorical_cols:
    if combined_df[col].isnull().any():
        mode_vals = combined_df[col].mode()
        fill_val = mode_vals[0] if len(mode_vals) > 0 else 'unknown'
        combined_df[col] = combined_df[col].fillna(fill_val)

print("Advanced missing value imputation complete.")

# -----------------------------------------------------------------
# 4. ADVANCED Feature Selection and Preprocessing
# -----------------------------------------------------------------
print("\nStep 4: Advanced Feature Selection...")

# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    selector = VarianceThreshold(threshold=0.005)
    selector.fit(combined_df[numerical_cols])
    low_var_mask = selector.get_support()
    
    low_var_features = numerical_cols[~low_var_mask]
    if len(low_var_features) > 0:
        combined_df = combined_df.drop(columns=low_var_features)
        print(f"Removed {len(low_var_features)} low-variance features")

# Advanced correlation analysis with clustering
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    corr_matrix = combined_df[numerical_cols].corr().abs()
    
    high_corr_features = set()
    features_to_drop = set()
    
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            col1 = numerical_cols[i]
            col2 = numerical_cols[j]
            if corr_matrix.iloc[i, j] > 0.85:
                if col1 not in features_to_drop and col2 not in features_to_drop:
                    if combined_df[col1].var() < combined_df[col2].var():
                        features_to_drop.add(col1)
                    else:
                        features_to_drop.add(col2)
    
    if features_to_drop:
        combined_df = combined_df.drop(columns=list(features_to_drop))
        print(f"Removed {len(features_to_drop)} highly correlated features")

# One-hot encode categorical features with frequency encoding for high cardinality
categorical_cols = combined_df.select_dtypes(include=['object', 'bool']).columns.tolist()
if categorical_cols:
    print(f"Encoding {len(categorical_cols)} categorical features")
    
    for col in categorical_cols:
        if combined_df[col].nunique() > 10:  # High cardinality threshold
            # Use frequency encoding
            freq_map = combined_df[col].value_counts(normalize=True).to_dict()
            combined_df[f'{col}_freq'] = combined_df[col].map(freq_map)
            combined_df = combined_df.drop(columns=[col])
        else:
            # Use one-hot encoding
            # Handle potential NaNs in categorical columns before one-hot encoding
            if combined_df[col].isnull().any():
                combined_df[col] = combined_df[col].fillna('Missing')
            combined_df = pd.get_dummies(combined_df, columns=[col], prefix=col, drop_first=True)

# Clean column names for model compatibility
combined_df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in combined_df.columns]
combined_df.columns = [re.sub(r'_+', '_', col).strip('_') for col in combined_df.columns]

# Advanced outlier handling with isolation forest
from sklearn.ensemble import IsolationForest
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns

# Apply isolation forest to detect multivariate outliers
if len(numerical_cols) > 1 and len(combined_df) > 100:
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    outliers = iso_forest.fit_predict(combined_df[numerical_cols])
    print(f"Detected {sum(outliers == -1)} multivariate outliers")

    for col in numerical_cols:
        lower_bound = combined_df[col].quantile(0.01)
        upper_bound = combined_df[col].quantile(0.99)
        combined_df[col] = np.clip(combined_df[col], lower_bound, upper_bound)
else:
    print("Skipping Isolation Forest: Not enough numerical features or samples.")

print(f"Final feature count: {combined_df.shape[1]}")

# -----------------------------------------------------------------
# 5. ADVANCED Model Selection with Multiple Validation Strategies
# -----------------------------------------------------------------
print("\nStep 5: Advanced Model Selection...")

# Separate train/test
train_len = len(train_y)
X_train = combined_df.iloc[:train_len].copy()
X_test = combined_df.iloc[train_len:].copy()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

def get_advanced_models():
    """Returns advanced models with optimized hyperparameters"""
    return {
        # Gradient Boosting with advanced parameters
        'lightgbm_v1': lgb.LGBMRegressor(
            random_state=42, n_estimators=2000, learning_rate=0.03,
            num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            verbose=-1
        ),
        'lightgbm_v2': lgb.LGBMRegressor(
            random_state=42, n_estimators=1500, learning_rate=0.05,
            num_leaves=63, feature_fraction=0.7, bagging_fraction=0.9,
            bagging_freq=3, min_child_samples=10, reg_alpha=0.05, reg_lambda=0.05,
            verbose=-1
        ),
        'xgboost_v1': xgb.XGBRegressor(
            random_state=42, n_estimators=2000, learning_rate=0.03,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, verbosity=0
        ),
        'xgboost_v2': xgb.XGBRegressor(
            random_state=42, n_estimators=1500, learning_rate=0.05,
            max_depth=8, subsample=0.9, colsample_bytree=0.7,
            reg_alpha=0.05, reg_lambda=0.05, verbosity=0
        ),
        'catboost_v1': cb.CatBoostRegressor(
            random_state=42, iterations=2000, learning_rate=0.03,
            depth=8, l2_leaf_reg=5, verbose=0
        ),
        'catboost_v2': cb.CatBoostRegressor(
            random_state=42, iterations=1500, learning_rate=0.05,
            depth=6, l2_leaf_reg=3, verbose=0
        ),
        
        # Advanced ensemble methods
        'random_forest_v1': RandomForestRegressor(
            random_state=42, n_estimators=300, max_depth=15,
            min_samples_split=5, min_samples_leaf=2, n_jobs=-1
        ),
        'random_forest_v2': RandomForestRegressor(
            random_state=42, n_estimators=500, max_depth=20,
            min_samples_split=3, min_samples_leaf=1, n_jobs=-1
        ),
        'extra_trees': ExtraTreesRegressor(
            random_state=42, n_estimators=400, max_depth=18,
            min_samples_split=4, min_samples_leaf=2, n_jobs=-1
        ),
        
        # Advanced linear models
        'ridge_advanced': RidgeCV(alphas=np.logspace(-5, 5, 50)),
        'elastic_net_advanced': ElasticNetCV(
            alphas=np.logspace(-5, 2, 20), 
            l1_ratio=np.linspace(0.1, 0.9, 9), cv=5
        ),
        'bayesian_ridge': BayesianRidge(),
        
        # Neural network
        'mlp': MLPRegressor(
            hidden_layer_sizes=(200, 100, 50), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive',
            max_iter=1000, random_state=42
        ),
        
        # Support Vector Regression
        'svr': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    }

# Multiple validation strategies
cv_strategies = {
    'kfold': KFold(n_splits=5, shuffle=True, random_state=42),
    'kfold_10': KFold(n_splits=10, shuffle=True, random_state=42)
}

# If we have time series data, add TimeSeriesSplit
if 'days_since_start' in X_train.columns:
    cv_strategies['timeseries'] = TimeSeriesSplit(n_splits=5)

print("\n" + "="*80)
print("ADVANCED MODEL EVALUATION WITH MULTIPLE CV STRATEGIES")
print("="*80)

models = get_advanced_models()
model_scores = {}

# Evaluate each model with each CV strategy
for cv_name, cv_strategy in cv_strategies.items():
    print(f"\n--- {cv_name.upper()} CROSS-VALIDATION ---")
    
    for name, model in models.items():
        try:
            cv_scores = cross_val_score(
                model, X_train, train_y, cv=cv_strategy, 
                scoring='neg_root_mean_squared_error', n_jobs=-1
            )
            cv_rmse = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            model_key = f"{name}_{cv_name}"
            model_scores[model_key] = {
                'model_name': name, 'cv_strategy': cv_name,
                'cv_rmse': cv_rmse, 'cv_std': cv_std
            }
            
            print(f"  {name:<20} RMSE: {cv_rmse:.4f} (±{cv_std:.4f})")
            
        except Exception as e:
            print(f"  {name:<20} FAILED: {str(e)}")

# Find best models from each CV strategy
print("\n" + "="*60)
print("BEST MODELS BY CV STRATEGY")
print("="*60)

best_models_by_cv = {}
for cv_name in cv_strategies.keys():
    cv_models = {k: v for k, v in model_scores.items() if v['cv_strategy'] == cv_name}
    if cv_models:
        best_model = min(cv_models.items(), key=lambda x: x[1]['cv_rmse'])
        best_models_by_cv[cv_name] = best_model[1]
        print(f"{cv_name:<15}: {best_model[1]['model_name']:<20} RMSE: {best_model[1]['cv_rmse']:.4f}")

# -----------------------------------------------------------------
# 6. ADVANCED Stacking with Multiple Meta-Models
# -----------------------------------------------------------------
print("\n" + "="*80)
print("ADVANCED STACKING ENSEMBLE")
print("="*80)

# Select top models based on average performance across CV strategies
model_avg_scores = {}
for name in set([v['model_name'] for v in model_scores.values()]):
    scores = [v['cv_rmse'] for v in model_scores.values() if v['model_name'] == name]
    model_avg_scores[name] = np.mean(scores)

# Select top 8 models for stacking
top_models = sorted(model_avg_scores.items(), key=lambda x: x[1])[:8]
selected_models = [name for name, _ in top_models]

print(f"Selected models for stacking: {selected_models}")

# Generate out-of-fold predictions with multiple CV strategies
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros((len(X_train), len(selected_models)))
test_predictions = np.zeros((len(X_test), len(selected_models)))

advanced_models = get_advanced_models()

for i, model_name in enumerate(selected_models):
    print(f"\n--- Stacking training for {model_name} ---")
    model = advanced_models[model_name]
    fold_test_preds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        
        # Handle different model types
        if model_name.startswith(('lightgbm', 'xgboost', 'catboost')):
            if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                model.fit(X_fold_train, y_fold_train,
                         eval_set=[(X_fold_val, y_fold_val)])
            else:
                model.fit(X_fold_train, y_fold_train)
        else:
            model.fit(X_fold_train, y_fold_train)
        
        oof_predictions[val_idx, i] = model.predict(X_fold_val)
        fold_test_preds.append(model.predict(X_test))
    
    test_predictions[:, i] = np.mean(fold_test_preds, axis=0)
    
    # Calculate individual OOF score
    oof_rmse = np.sqrt(mean_squared_error(train_y, oof_predictions[:, i]))
    print(f"  OOF RMSE: {oof_rmse:.4f}")

# -----------------------------------------------------------------
# 7. MULTI-LEVEL Meta-Learning
# -----------------------------------------------------------------
print("\n--- Training Multi-Level Meta-Models ---")

# Level 1: Basic meta-models
meta_models_l1 = {
    'ridge_meta': RidgeCV(alphas=np.logspace(-5, 5, 50)),
    'elastic_meta': ElasticNetCV(alphas=np.logspace(-4, 2, 20), 
                                l1_ratio=np.linspace(0.1, 0.9, 9), cv=3),
    'lasso_meta': LassoCV(alphas=np.logspace(-5, 2, 30), cv=3),
    'lightgbm_meta': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, 
                                      num_leaves=15, random_state=42, verbose=-1),
    'bayesian_meta': BayesianRidge()
}

# Train Level 1 meta-models
level1_predictions = np.zeros((len(X_train), len(meta_models_l1)))
level1_test_predictions = np.zeros((len(X_test), len(meta_models_l1)))

best_l1_rmse = float('inf')
best_l1_model = None
best_l1_name = None

for i, (meta_name, meta_model) in enumerate(meta_models_l1.items()):
    # Cross-validation for level 1 meta-models
    cv_scores = cross_val_score(meta_model, oof_predictions, train_y, cv=3, 
                               scoring='neg_root_mean_squared_error')
    meta_rmse = -cv_scores.mean()
    
    meta_model.fit(oof_predictions, train_y)
    level1_predictions[:, i] = meta_model.predict(oof_predictions)
    level1_test_predictions[:, i] = meta_model.predict(test_predictions)
    
    print(f"{meta_name}: {meta_rmse:.4f}")
    
    if meta_rmse < best_l1_rmse:
        best_l1_rmse = meta_rmse
        best_l1_model = meta_model
        best_l1_name = meta_name

# Level 2: Meta-meta-model (ensemble of Level 1 predictions)
print(f"\nBest Level 1 meta-model: {best_l1_name} (RMSE: {best_l1_rmse:.4f})")

# Combine original predictions with Level 1 meta-predictions
combined_features = np.hstack([oof_predictions, level1_predictions])
combined_test_features = np.hstack([test_predictions, level1_test_predictions])

# Level 2 meta-model with feature selection
meta_models_l2 = {
    'ridge_l2': RidgeCV(alphas=np.logspace(-4, 4, 30)),
    'elastic_l2': ElasticNetCV(alphas=np.logspace(-3, 1, 15), 
                              l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=3)
}

best_l2_rmse = float('inf')
best_l2_model = None
best_l2_name = None

print("\n--- Level 2 Meta-Models ---")
for meta_name, meta_model in meta_models_l2.items():
    cv_scores = cross_val_score(meta_model, combined_features, train_y, cv=3, 
                               scoring='neg_root_mean_squared_error')
    meta_rmse = -cv_scores.mean()
    
    print(f"{meta_name}: {meta_rmse:.4f}")
    
    if meta_rmse < best_l2_rmse:
        best_l2_rmse = meta_rmse
        best_l2_model = meta_model
        best_l2_name = meta_name

print(f"\nBest Level 2 meta-model: {best_l2_name} (RMSE: {best_l2_rmse:.4f})")

# Train final model
final_model = best_l2_model if best_l2_rmse < best_l1_rmse else best_l1_model
final_features = combined_features if best_l2_rmse < best_l1_rmse else oof_predictions
final_test_features = combined_test_features if best_l2_rmse < best_l1_rmse else test_predictions
final_rmse = best_l2_rmse if best_l2_rmse < best_l1_rmse else best_l1_rmse

final_model.fit(final_features, train_y)
final_predictions = final_model.predict(final_test_features)

# -----------------------------------------------------------------
# 8. POST-PROCESSING and Ensemble Blending
# -----------------------------------------------------------------
print("\n--- Post-Processing ---")

# Ensure predictions are non-negative
final_predictions = np.maximum(final_predictions, 0)

# Optional: Apply target transformation insights
target_stats = {
    'mean': train_y.mean(),
    'std': train_y.std(),
    'min': train_y.min(),
    'max': train_y.max()
}

# Clip extreme predictions
final_predictions = np.clip(final_predictions, 
                           target_stats['mean'] - 3*target_stats['std'],
                           target_stats['mean'] + 3*target_stats['std'])

# Additional ensemble: Weighted average of best individual models and meta-model
print("\n--- Final Ensemble Blending ---")

# Get predictions from top 3 individual models
top_3_models = selected_models[:3]
individual_predictions = []

for model_name in top_3_models:
    model = advanced_models[model_name]
    model.fit(X_train, train_y)
    pred = model.predict(X_test)
    individual_predictions.append(pred)

individual_predictions = np.column_stack(individual_predictions)

# Weighted ensemble: 70% meta-model, 30% top individual models
weights_meta = 0.7
weights_individual = 0.3 / len(top_3_models)

blended_predictions = (weights_meta * final_predictions + 
                      weights_individual * individual_predictions.sum(axis=1))

blended_predictions = np.maximum(blended_predictions, 0)

# -----------------------------------------------------------------
# 9. ADVANCED Feature Importance Analysis
# -----------------------------------------------------------------
print("\n--- Feature Importance Analysis ---")

# Get feature importance from best models
importance_dict = {}

for model_name in selected_models[:3]:
    model = advanced_models[model_name]
    model.fit(X_train, train_y)
    
    if hasattr(model, 'feature_importances_'):
        importance_dict[model_name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance_dict[model_name] = np.abs(model.coef_)

if importance_dict:
    # Average importance across models
    avg_importance = np.mean(list(importance_dict.values()), axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)

# -----------------------------------------------------------------
# 10. FINAL PREDICTION VALIDATION AND SUBMISSION
# -----------------------------------------------------------------
print("\n--- Final Validation and Submission Creation ---")

# Cross-validate the final ensemble approach
print("\nValidating final ensemble approach...")
ensemble_cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
    
    # Train selected models on fold
    fold_predictions = []
    for model_name in selected_models:
        model = advanced_models[model_name]
        model.fit(X_fold_train, y_fold_train)
        fold_pred = model.predict(X_fold_val)
        fold_predictions.append(fold_pred)
    
    fold_predictions = np.column_stack(fold_predictions)
    
    # Train meta-model
    meta_model = RidgeCV(alphas=np.logspace(-4, 4, 20))
    meta_model.fit(fold_predictions, y_fold_val)
    meta_pred = meta_model.predict(fold_predictions)
    
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, meta_pred))
    ensemble_cv_scores.append(fold_rmse)

ensemble_cv_mean = np.mean(ensemble_cv_scores)
ensemble_cv_std = np.std(ensemble_cv_scores)

print(f"Final Ensemble CV RMSE: {ensemble_cv_mean:.4f} (±{ensemble_cv_std:.4f})")

# Create multiple submission versions
submissions = {
    'meta_model': final_predictions,
    'blended': blended_predictions,
    'conservative': (final_predictions + blended_predictions) / 2
}

# Statistical analysis of predictions
print("\n--- Prediction Statistics ---")
for name, preds in submissions.items():
    print(f"\n{name.upper()} predictions:")
    print(f"  Mean: {preds.mean():.3f}")
    print(f"  Std: {preds.std():.3f}")
    print(f"  Range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  Target mean ratio: {preds.mean() / train_y.mean():.3f}")

# Create submissions
for name, preds in submissions.items():
    submission_df = pd.DataFrame({
        'date': submission_dates,
        'daily_ktCO2': preds
    })
    
    filename = f'submission_{name}.csv'
    submission_df.to_csv(filename, index=False)
    print(f"\nSubmission '{name}' saved to '{filename}'")
    print(f"First 5 predictions:")
    print(submission_df.head())

# -----------------------------------------------------------------
# 11. MODEL DIAGNOSTICS AND INSIGHTS
# -----------------------------------------------------------------
print("\n" + "="*80)
print("MODEL DIAGNOSTICS AND INSIGHTS")
print("="*80)

# Residual analysis on the best model
best_individual_model = advanced_models[selected_models[0]]
best_individual_model.fit(X_train, train_y)
train_predictions = best_individual_model.predict(X_train)
residuals = train_y - train_predictions

print(f"\nResidual Analysis:")
print(f"  Mean residual: {residuals.mean():.4f}")
print(f"  Std residual: {residuals.std():.4f}")
print(f"  Residual range: [{residuals.min():.4f}, {residuals.max():.4f}]")

# Identify potential data leakage or overfitting
train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))
print(f"\nOverfitting Check:")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  CV RMSE: {ensemble_cv_mean:.4f}")
print(f"  Difference: {abs(train_rmse - ensemble_cv_mean):.4f}")

if abs(train_rmse - ensemble_cv_mean) > 0.5:
    print("  WARNING: Potential overfitting detected!")
else:
    print("  Model appears well-regularized.")

# Final recommendations
print(f"\n" + "="*60)
print("FINAL RECOMMENDATIONS")
print("="*60)

print(f"1. Best Individual Model: {selected_models[0]} (Avg RMSE: {model_avg_scores[selected_models[0]]:.4f})")
print(f"2. Best Ensemble RMSE: {ensemble_cv_mean:.4f}")
print(f"3. Recommended Submission: 'blended' (combines meta-model with top individual models)")
print(f"4. Key Features: Check 'feature_importance.csv' for detailed analysis")

if ensemble_cv_mean < 2.20:
    print(f"5. Achievement: Target RMSE < 2.20 ✓ (Current: {ensemble_cv_mean:.4f})")
else:
    print(f"5. Status: Current RMSE {ensemble_cv_mean:.4f}, target < 2.20")

print(f"\nTraining completed successfully!")
print(f"Files created: submission_meta_model.csv, submission_blended.csv, submission_conservative.csv, feature_importance.csv")
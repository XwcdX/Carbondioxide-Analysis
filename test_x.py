import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.neural_network import MLPRegressor
import re
import warnings
import os
import random
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

print("Step 1: Loading and Cleaning Data...")

train_df = pd.read_csv('datatrain.csv')
test_df = pd.read_csv('datatest.csv')
submission_dates = test_df['date']
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Store ids and target
train_ids = train_df['obs_id']
test_ids = test_df['obs_id']
train_y = train_df['daily_ktCO2']

print(f"Target stats - Mean: {train_y.mean():.3f}, Std: {train_y.std():.3f}")
print(f"Target range: [{train_y.min():.3f}, {train_y.max():.3f}]")

# Remove target and id columns
train_df = train_df.drop(columns=['daily_ktCO2', 'obs_id'])
test_df = test_df.drop(columns=['obs_id'])

# Combine datasets
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Data preprocessing (keeping your existing preprocessing)
text_cols = combined_df.select_dtypes(include=['object']).columns
for col in text_cols:
    if col != 'date':
        combined_df[col] = combined_df[col].astype(str).str.lower().str.strip()

bool_cols = ['rain', 'weekend', 'holiday']
for col in bool_cols:
    if col in combined_df.columns:
        map_dict = {'true': "yes", 'yes': "yes", '1': "yes", 'false': "no", 'no': "no", '0': "no"}
        combined_df[col] = combined_df[col].astype(str).str.strip().str.lower().map(map_dict)

combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['so2'] = pd.to_numeric(combined_df['so2'], errors='coerce')

# Handle missing values
missing_analysis = pd.DataFrame({
    'column': combined_df.columns,
    'missing_count': combined_df.isnull().sum(),
    'missing_pct': (combined_df.isnull().sum() / len(combined_df)) * 100
})
missing_analysis = missing_analysis[missing_analysis['missing_count'] > 0].sort_values('missing_pct', ascending=False)

high_missing_cols = missing_analysis[missing_analysis['missing_pct'] > 70]['column'].tolist()
if high_missing_cols:
    combined_df = combined_df.drop(columns=high_missing_cols)
    print(f"\nDropped {len(high_missing_cols)} columns with >70% missing data")

# Seasonal imputation
combined_df["doy"] = combined_df["date"].dt.dayofyear
for col in ['no2', 'o3', 'TCI', 'so2']:
    if col in combined_df.columns:
        seasonal_avg = combined_df.groupby("doy")[col].mean()
        combined_df[f"{col}_imputed"] = combined_df.apply(
            lambda row: seasonal_avg[row["doy"]] if pd.isna(row[col]) else row[col],
            axis=1
        )

combined_df['so2_imputed'] = combined_df['so2_imputed'].interpolate()

# Feature engineering (keeping your existing features)
combined_df['month'] = combined_df['date'].dt.month
combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
combined_df['day_of_year'] = combined_df['date'].dt.dayofyear
combined_df['week_of_year'] = combined_df['date'].dt.isocalendar().week.astype(int)
combined_df['year'] = combined_df['date'].dt.year
combined_df['quarter'] = combined_df['date'].dt.quarter
combined_df['is_month_start'] = combined_df['date'].dt.is_month_start.astype(int)
combined_df['is_month_end'] = combined_df['date'].dt.is_month_end.astype(int)

combined_df['weekend'] = combined_df['day_of_week'].apply(lambda x: 'yes' if x >= 5 else 'no')

import holidays
indo_holidays = holidays.CountryHoliday('ID')
combined_df['holiday'] = combined_df['date'].isin(indo_holidays).map({True: 'yes', False: 'no'})

def add_cyclical_features(df, col, period):
    if col in df.columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df

if 'winddir' in combined_df.columns:
    combined_df = add_cyclical_features(combined_df, 'winddir', 360)
    combined_df = combined_df.drop('winddir', axis=1)

combined_df = add_cyclical_features(combined_df, 'month', 12)
combined_df = add_cyclical_features(combined_df, 'day_of_week', 7)
combined_df = add_cyclical_features(combined_df, 'day_of_year', 365)
combined_df = add_cyclical_features(combined_df, 'week_of_year', 52)
combined_df = add_cyclical_features(combined_df, 'quarter', 4)

# Continue with all your feature engineering...
weather_cols = ['temp', 'tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure']
pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2']

if all(col in combined_df.columns for col in ['temp', 'tempmax', 'tempmin']):
    combined_df['temp_range'] = combined_df['tempmax'] - combined_df['tempmin']
    combined_df['temp_variance'] = combined_df['temp_range'] / (combined_df['temp'].replace(0, 1e-6) + 1e-6)

if 'temp' in combined_df.columns and 'humidity' in combined_df.columns:
    combined_df['heat_index'] = combined_df['temp'] * (1 + combined_df['humidity'] / 100)
    combined_df['comfort_index'] = combined_df['temp'] * (1 - abs(combined_df['humidity'] - 50) / 100)

# Continue with rest of feature engineering...
wind_features = ['windspeed', 'windgust', 'windspeedmax', 'windspeedmin']
for feat in wind_features:
    if feat in combined_df.columns and 'temp' in combined_df.columns:
        combined_df[f'{feat}_temp_interaction'] = combined_df[feat] * combined_df['temp']

if 'sealevelpressure' in combined_df.columns:
    combined_df['pressure_anomaly'] = combined_df['sealevelpressure'] - combined_df['sealevelpressure'].mean()

# Pollutant features
if len([col for col in pollutant_cols if col in combined_df.columns]) > 1:
    available_pollutants = [col for col in pollutant_cols if col in combined_df.columns]
    combined_df['pollutant_sum'] = combined_df[available_pollutants].sum(axis=1)
    combined_df['pollutant_mean'] = combined_df[available_pollutants].mean(axis=1)
    
    if 'pm25' in combined_df.columns and 'pm10' in combined_df.columns:
        combined_df['fine_coarse_ratio'] = combined_df['pm25'] / (combined_df['pm10'] + 1e-6)
    
    if 'no2' in combined_df.columns and 'o3' in combined_df.columns:
        combined_df['no2_o3_ratio'] = combined_df['no2_imputed'] / (combined_df['o3_imputed'] + 1e-6)

# Handle categorical variables
categorical_cols = combined_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    combined_df[col] = combined_df[col].replace("nan", np.nan)
    if combined_df[col].isnull().any():
        mode_vals = combined_df[col].mode()
        fill_val = mode_vals[0] if len(mode_vals) > 0 else 'unknown'
        combined_df[col] = combined_df[col].fillna(fill_val)

# Drop some columns and create new ones
cyclical_to_drop = ['month', 'day_of_week', 'day_of_year', 'week_of_year', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
combined_df = combined_df.drop(columns=[col for col in cyclical_to_drop if col in combined_df.columns])

combined_df['pm10_log'] = np.log1p(combined_df['pm10'])
combined_df = combined_df.drop(['pm10'], axis=1)

oh_encode_cols = ['rain', 'conditions', 'seasons', 'weekend', 'holiday']
combined_df = pd.get_dummies(combined_df, columns=oh_encode_cols, drop_first=True)

combined_df['pm_ratio'] = combined_df['pm25'] / (combined_df['pm10_log'])

bool_cols = combined_df.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    combined_df[col] = combined_df[col].astype(int)

combined_df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in combined_df.columns]
combined_df.columns = [re.sub(r'_+', '_', col).strip('_') for col in combined_df.columns]

combined_df = combined_df.sort_values('date')

# Lag and rolling features
features_to_lag = ['pm25', 'temp', 'humidity', 'windspeed', 'o3_imputed', 'no2_imputed', 'pollutant_sum']
lag_periods = [1, 2, 3, 7, 14]
rolling_windows = [3, 7, 14]

for col in features_to_lag:
    if col in combined_df.columns:
        for lag in lag_periods:
            combined_df[f'{col}_lag_{lag}'] = combined_df[col].shift(lag)
        
        for window in rolling_windows:
            combined_df[f'{col}_roll_mean_{window}'] = combined_df[col].shift(1).rolling(window=window).mean()
            combined_df[f'{col}_roll_std_{window}'] = combined_df[col].shift(1).rolling(window=window).std()

# Fill missing values
combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
for col in combined_df.select_dtypes(include=[np.number]).columns:
    if combined_df[col].isna().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Clean up columns
combined_df = combined_df.drop([col for col in ['o3', 'no2', 'so2', 'TCI'] if col in combined_df.columns], axis=1)

# Prepare final datasets
X_train = combined_df.iloc[train_ids-1, :].drop(['date'], axis=1)
X_test = combined_df.iloc[test_ids-1, :].drop(['date'], axis=1)

print(f"Final feature shape: {X_train.shape}")

# Fixed Sequential Models
class FixedSequentialFeatureExtractor:
    def __init__(self, seq_length=7, feature_names=None):
        self.seq_length = seq_length
        self.feature_names = feature_names
        self.scalers = {}
        
    def create_sequences(self, X, y=None):
        if len(X) <= self.seq_length:
            # Return original data if not enough samples
            return X, y
            
        sequences = []
        targets = [] if y is not None else None
        
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if (y is not None and hasattr(y, 'values')) else y
        
        for i in range(self.seq_length, len(X_array)):
            seq = X_array[i-self.seq_length:i]
            
            # Extract statistical features from sequence
            seq_features = []
            seq_features.extend(np.mean(seq, axis=0))  # Mean
            seq_features.extend(np.std(seq, axis=0))   # Std
            seq_features.extend(seq[-1])  # Most recent values
            
            # Trend features
            for j in range(seq.shape[1]):
                x_vals = np.arange(self.seq_length)
                y_vals = seq[:, j]
                if np.std(y_vals) > 1e-6:
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                else:
                    slope = 0
                seq_features.append(slope)
            
            sequences.append(seq_features)
            if y is not None:
                if targets is None:
                    targets = []
                targets.append(y_array[i] if y_array is not None else 0)
                
        return np.array(sequences), np.array(targets) if targets is not None else None

class FixedSequentialWrapper:
    def __init__(self, base_model, seq_length=7):
        self.base_model = base_model
        self.seq_length = seq_length
        self.feature_extractor = FixedSequentialFeatureExtractor(seq_length)
        self.use_sequential = False
        self.fallback_prediction = 0
        
    def fit(self, X, y):
        if len(X) <= self.seq_length:
            # Use regular model for small datasets
            self.base_model.fit(X, y)
            self.use_sequential = False
            self.fallback_prediction = np.mean(y)
        else:
            try:
                X_seq, y_seq = self.feature_extractor.create_sequences(X, y)
                if len(X_seq) > 0:
                    self.base_model.fit(X_seq, y_seq)
                    self.use_sequential = True
                    self.fallback_prediction = np.mean(y_seq)
                else:
                    self.base_model.fit(X, y)
                    self.use_sequential = False
                    self.fallback_prediction = np.mean(y)
            except Exception as e:
                print(f"Sequential fit failed: {e}, using regular model")
                self.base_model.fit(X, y)
                self.use_sequential = False
                self.fallback_prediction = np.mean(y)
        return self
    
    def predict(self, X):
        if not self.use_sequential or len(X) <= self.seq_length:
            return self.base_model.predict(X)
        else:
            try:
                X_seq, _ = self.feature_extractor.create_sequences(X, None)
                if len(X_seq) > 0:
                    pred_seq = self.base_model.predict(X_seq)
                    full_pred = np.full(len(X), self.fallback_prediction)
                    full_pred[self.seq_length:] = pred_seq
                    return full_pred
                else:
                    return self.base_model.predict(X)
            except Exception as e:
                print(f"Sequential predict failed: {e}, using regular model")
                return self.base_model.predict(X)

# Enhanced Neural Network
class EnhancedMLPRegressor:
    def __init__(self, hidden_layer_sizes=(200, 100, 50), alpha=0.01, max_iter=500):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate='adaptive',
            solver='adam',
            random_state=SEED
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def optimize_lightgbm(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = lgb.LGBMRegressor(
        **params,
        n_estimators=1000,
        random_state=SEED,
        verbose=-1
    )
    
    tsfold = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()

def optimize_xgboost(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = xgb.XGBRegressor(
        **params,
        n_estimators=1000,
        random_state=SEED,
        verbosity=0
    )
    
    tsfold = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()

def optimize_catboost(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'early_stopping_rounds': 150,
        'border_count': trial.suggest_int('border_count', 64, 255),
    }
    
    model = cb.CatBoostRegressor(
        **params,
        random_state=SEED,
        verbose=0,
        task_type='GPU',
        devices='0',
        bootstrap_type='Bayesian',
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.5, 1.0),
        od_type='Iter',
        metric_period=100,
        gpu_ram_part=0.5,
        max_ctr_complexity=4,
    )
    
    tsfold = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=1)
    return -cv_scores.mean()

def optimize_random_forest(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    
    model = RandomForestRegressor(
        **params,
        random_state=SEED,
        n_jobs=-1
    )
    
    tsfold = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()

def optimize_gradient_boosting(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    
    model = GradientBoostingRegressor(
        **params,
        random_state=SEED
    )
    
    tsfold = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()

def optimize_extra_trees(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    
    model = ExtraTreesRegressor(
        **params,
        random_state=SEED,
        n_jobs=-1
    )
    
    tsfold = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()

print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
print("="*60)

optuna.logging.set_verbosity(optuna.logging.WARNING)


best_params = {
    # old
    # 'lightgbm': { # 1.7049
    #     'learning_rate': 0.05127862283124463, 
    #     'num_leaves': 245, 
    #     'max_depth': 10, 
    #     'min_child_samples': 23, 
    #     'subsample': 0.6176714838282072, 
    #     'colsample_bytree': 0.878721519175876, 
    #     'reg_alpha': 9.923772567588348e-06, 
    #     'reg_lambda': 2.76835469353685
    # },
    # 'xgboost': { # 1.6238
    #     'learning_rate': 0.08204180423731024, 
    #     'max_depth': 7, 
    #     'subsample': 0.920817137596488, 
    #     'colsample_bytree': 0.8160571823010057, 
    #     'reg_alpha': 3.561302339114832e-07, 
    #     'reg_lambda': 2.3257577031009327e-05
    # },
    # 'catboost': { #1.5590
    #     'learning_rate': 0.0585971302788046, 
    #     'depth': 10, 'l2_leaf_reg': 0.010842262717330166, 
    #     'iterations': 1163, 
    #     'border_count': 123, 
    #     'bagging_temperature': 0.7600340105889054
    # },
    # 'random_forest':{ #1.9617
    #     'n_estimators': 340, 
    #     'max_depth': 22, 
    #     'min_samples_split': 2, 
    #     'min_samples_leaf': 2, 
    #     'max_features': None
    # },
    # 'gradient_boosting':{ #1.7262
    #     'n_estimators': 461, 
    #     'learning_rate': 0.018254861693279043, 
    #     'max_depth': 7, 
    #     'subsample': 0.8631663038344568
    # },
    # 'extra_trees':{ #1.5225
    #     'n_estimators': 420, 
    #     'max_depth': 22, 
    #     'min_samples_split': 3, 
    #     'min_samples_leaf': 2, 
    #     'max_features': None
    # }
}

# Optuna parameter tuning
optimizers = {
    'lightgbm': optimize_lightgbm,
    'xgboost': optimize_xgboost,
    'catboost': optimize_catboost,
    'random_forest': optimize_random_forest,
    'gradient_boosting': optimize_gradient_boosting,
    'extra_trees': optimize_extra_trees,
}

N_TRIALS = 150 
CATBOOST_TRIALS = 150

for model_name, optimizer in optimizers.items():
    trials = CATBOOST_TRIALS if model_name == 'catboost' else N_TRIALS
    print(f"\nOptimizing {model_name} with {trials} trials...")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    
    if model_name == 'catboost':
        print("  (CatBoost optimization may take longer, please wait...)")
        study.optimize(optimizer, n_trials=trials, timeout=1800)
    else:
        study.optimize(optimizer, n_trials=trials, show_progress_bar=True)
    
    best_params[model_name] = study.best_params
    print(f"Best {model_name} RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

class FixedSequentialFeatureExtractor(BaseEstimator):
    def __init__(self, seq_length=7, feature_names=None):
        self.seq_length = seq_length
        self.feature_names = feature_names
        self.scalers = {}
        
    def get_params(self, deep=True):
        return {
            'seq_length': self.seq_length,
            'feature_names': self.feature_names
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
        
    def create_sequences(self, X, y=None):
        if len(X) <= self.seq_length:
            return X, y
            
        sequences = []
        targets = [] if y is not None else None
        
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if (y is not None and hasattr(y, 'values')) else y
        
        for i in range(self.seq_length, len(X_array)):
            seq = X_array[i-self.seq_length:i]
            
            # Extract statistical features from sequence
            seq_features = []
            seq_features.extend(np.mean(seq, axis=0))  # Mean
            seq_features.extend(np.std(seq, axis=0))   # Std
            seq_features.extend(seq[-1])  # Most recent values
            
            # Trend features
            for j in range(seq.shape[1]):
                x_vals = np.arange(self.seq_length)
                y_vals = seq[:, j]
                if np.std(y_vals) > 1e-6:
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                else:
                    slope = 0
                seq_features.append(slope)
            
            sequences.append(seq_features)
            if y is not None:
                if targets is None:
                    targets = []
                targets.append(y_array[i] if y_array is not None else 0)
                
        return np.array(sequences), np.array(targets) if targets is not None else None

class FixedSequentialWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, base_model_type='rf', seq_length=7, **base_model_params):
        self.base_model_type = base_model_type
        self.seq_length = seq_length
        self.base_model_params = base_model_params
        self.feature_extractor = FixedSequentialFeatureExtractor(seq_length)
        self.use_sequential = False
        self.fallback_prediction = 0
        self._fitted_model = None
        
    def _create_base_model(self):
        """Create base model based on type and parameters"""
        if self.base_model_type == 'rf':
            default_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'random_state': SEED,
                'n_jobs': -1
            }
            default_params.update(self.base_model_params)
            return RandomForestRegressor(**default_params)
        
        elif self.base_model_type == 'mlp':
            default_params = {
                'hidden_layer_sizes': (100, 50),
                'alpha': 0.01,
                'max_iter': 500,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'learning_rate': 'adaptive',
                'solver': 'adam',
                'random_state': SEED
            }
            default_params.update(self.base_model_params)
            return MLPRegressor(**default_params)
        
        else:
            raise ValueError(f"Unknown base_model_type: {self.base_model_type}")
        
    def get_params(self, deep=True):
        params = {
            'base_model_type': self.base_model_type,
            'seq_length': self.seq_length,
        }
        # Add base model parameters with prefix
        for key, value in self.base_model_params.items():
            params[key] = value
        return params
    
    def set_params(self, **params):
        base_model_params = {}
        for key, value in params.items():
            if key in ['base_model_type', 'seq_length']:
                setattr(self, key, value)
            else:
                base_model_params[key] = value
        
        self.base_model_params.update(base_model_params)
        # Recreate feature extractor with new seq_length
        self.feature_extractor = FixedSequentialFeatureExtractor(self.seq_length)
        return self
        
    def fit(self, X, y):
        # Create fresh model instance for fitting
        self._fitted_model = self._create_base_model()
        
        if len(X) <= self.seq_length:
            self._fitted_model.fit(X, y)
            self.use_sequential = False
            self.fallback_prediction = np.mean(y)
        else:
            try:
                X_seq, y_seq = self.feature_extractor.create_sequences(X, y)
                if len(X_seq) > 0:
                    # Scale features for MLP
                    if self.base_model_type == 'mlp':
                        self.scaler = StandardScaler()
                        X_seq = self.scaler.fit_transform(X_seq)
                    
                    self._fitted_model.fit(X_seq, y_seq)
                    self.use_sequential = True
                    self.fallback_prediction = np.mean(y_seq)
                else:
                    if self.base_model_type == 'mlp':
                        self.scaler = StandardScaler()
                        X_scaled = self.scaler.fit_transform(X)
                        self._fitted_model.fit(X_scaled, y)
                    else:
                        self._fitted_model.fit(X, y)
                    self.use_sequential = False
                    self.fallback_prediction = np.mean(y)
            except Exception as e:
                print(f"Sequential fit failed: {e}, using regular model")
                if self.base_model_type == 'mlp':
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                    self._fitted_model.fit(X_scaled, y)
                else:
                    self._fitted_model.fit(X, y)
                self.use_sequential = False
                self.fallback_prediction = np.mean(y)
        return self
    
    def predict(self, X):
        if self._fitted_model is None:
            raise ValueError("Model not fitted yet")
            
        if not self.use_sequential or len(X) <= self.seq_length:
            if self.base_model_type == 'mlp' and hasattr(self, 'scaler'):
                X_scaled = self.scaler.transform(X)
                return self._fitted_model.predict(X_scaled)
            else:
                return self._fitted_model.predict(X)
        else:
            try:
                X_seq, _ = self.feature_extractor.create_sequences(X, None)
                if len(X_seq) > 0:
                    if self.base_model_type == 'mlp':
                        X_seq = self.scaler.transform(X_seq)
                    pred_seq = self._fitted_model.predict(X_seq)
                    full_pred = np.full(len(X), self.fallback_prediction)
                    full_pred[self.seq_length:] = pred_seq
                    return full_pred
                else:
                    if self.base_model_type == 'mlp' and hasattr(self, 'scaler'):
                        X_scaled = self.scaler.transform(X)
                        return self._fitted_model.predict(X_scaled)
                    else:
                        return self._fitted_model.predict(X)
            except Exception as e:
                print(f"Sequential predict failed: {e}, using regular model")
                if self.base_model_type == 'mlp' and hasattr(self, 'scaler'):
                    X_scaled = self.scaler.transform(X)
                    return self._fitted_model.predict(X_scaled)
                else:
                    return self._fitted_model.predict(X)

class EnhancedMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(200, 100, 50), alpha=0.01, max_iter=500, 
                 early_stopping=True, validation_fraction=0.1, learning_rate='adaptive', 
                 solver='adam', random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.learning_rate = learning_rate
        self.solver = solver
        self.random_state = random_state
        self.model = None
        self.scaler = None
        
    def get_params(self, deep=True):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'learning_rate': self.learning_rate,
            'solver': self.solver,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def fit(self, X, y):
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            learning_rate=self.learning_rate,
            solver=self.solver,
            random_state=self.random_state
        )
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        if self.model is None or self.scaler is None:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
# Get optimized models
def get_optimized_models():
    models = {
        'lightgbm': lgb.LGBMRegressor(
            **best_params['lightgbm'],
            n_estimators=1000,
            random_state=SEED,
            verbose=-1
        ),
        'xgboost': xgb.XGBRegressor(
            **best_params['xgboost'],
            n_estimators=1000,
            random_state=SEED,
            verbosity=0
        ),
        'catboost': cb.CatBoostRegressor(
            **best_params['catboost'],
            random_state=SEED,
            verbose=0,
            task_type='GPU',
            devices='0',
            bootstrap_type='Bayesian',
            od_type='Iter',
            gpu_ram_part=0.7,
            max_ctr_complexity=4
        ),
        'random_forest': RandomForestRegressor(
            **best_params['random_forest'],
            random_state=SEED,
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesRegressor(
            **best_params['extra_trees'],
            random_state=SEED,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            **best_params['gradient_boosting'],
            random_state=SEED
        ),
        'ridge': RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5),
        'elastic_net': ElasticNetCV(alphas=np.logspace(-4, 4, 10), 
                                   l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5),
        'lasso': LassoCV(alphas=np.logspace(-4, 4, 20), cv=5),
        
        'sequential_rf': FixedSequentialWrapper(
            base_model_type='rf',
            seq_length=7,
            n_estimators=200,
            max_depth=15,
            random_state=SEED,
            n_jobs=-1
        ),
        'sequential_mlp': FixedSequentialWrapper(
            base_model_type='mlp',
            seq_length=7,
            hidden_layer_sizes=(100, 50),
            alpha=0.01,
            max_iter=300,
            random_state=SEED
        ),
        
        # Enhanced Neural Networks
        'advanced_mlp': EnhancedMLPRegressor(
            hidden_layer_sizes=(200, 100, 50), 
            alpha=0.001,
            max_iter=500,
            random_state=SEED
        ),
    }
    return models

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

models = get_optimized_models()
model_scores = {}
model_predictions = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    try:
        if 'sequential' in name:
            temp_kfold = KFold(n_splits=3, shuffle=True, random_state=SEED)
            cv_scores = cross_val_score(model, X_train, train_y, cv=temp_kfold, 
                                       scoring='neg_root_mean_squared_error', n_jobs=1)
        else:
            cv_scores = cross_val_score(model, X_train, train_y, cv=kfold, 
                                       scoring='neg_root_mean_squared_error', n_jobs=-1)
        
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        model_scores[name] = {'cv_rmse': cv_rmse, 'cv_std': cv_std}
        
        model.fit(X_train, train_y)
        test_pred = model.predict(X_test)
        model_predictions[name] = test_pred
        
        print(f"  CV RMSE: {cv_rmse:.4f} (±{cv_std:.4f})")
        
    except Exception as e:
        print(f"  Error with {name}: {e}")
        model_scores[name] = {'cv_rmse': float('inf'), 'cv_std': 0}
        model_predictions[name] = np.zeros(len(X_test))

# Print results
print(f"\n{'Model':<18} {'CV RMSE':<10} {'Std':<10}")
print("-" * 40)
for name, scores in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse']):
    print(f"{name:<18} {scores['cv_rmse']:<10.4f} {scores['cv_std']:<10.4f}")

# Select top models for stacking
print("\n" + "="*60)
print("ENSEMBLE STACKING")
print("="*60)

selected_model_names = [name for name, _ in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[:7]]
print(f"Selected top 5 models: {selected_model_names}")

# Out-of-fold predictions for stacking
oof_predictions = np.zeros((len(X_train), len(selected_model_names)))
test_predictions = np.zeros((len(X_test), len(selected_model_names)))

models_for_stacking = get_optimized_models()

for i, name in enumerate(selected_model_names):
    print(f"--- Stacking training for {name} ---")
    model = models_for_stacking[name]
    fold_test_preds = []
    
    # Generate out-of-fold predictions
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_ids], X_train.iloc[val_ids]
        y_fold_train, y_fold_val = train_y.iloc[train_ids], train_y.iloc[val_ids]
        
        try:
            model.fit(X_fold_train, y_fold_train)
            val_pred = model.predict(X_fold_val)
            oof_predictions[val_ids, i] = val_pred
        except Exception as e:
            print(f"    Fold {fold} error: {e}")
            oof_predictions[val_ids, i] = np.mean(y_fold_train)

    # Generate test predictions using multiple folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
        X_fold_train = X_train.iloc[train_ids]
        y_fold_train = train_y.iloc[train_ids]
        
        try:
            model.fit(X_fold_train, y_fold_train)
            test_pred = model.predict(X_test)
            fold_test_preds.append(test_pred)
        except Exception as e:
            print(f"    Test fold {fold} error: {e}")
            fold_test_preds.append(np.full(len(X_test), np.mean(y_fold_train)))
    
    test_predictions[:, i] = np.mean(fold_test_preds, axis=0)

print("\nTraining meta-models...")
valid_oof_mask = np.any(oof_predictions != 0, axis=1)
valid_oof_indices = np.where(valid_oof_mask)[0]

print(f"Using {len(valid_oof_indices)} samples for meta-model training out of {len(X_train)} total")

oof_for_meta = oof_predictions[valid_oof_indices]
y_for_meta = train_y.iloc[valid_oof_indices]

# Optimize meta-models with Optuna
def optimize_meta_ridge(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
    model = RidgeCV(alphas=[alpha], cv=3)
    model.fit(oof_for_meta, y_for_meta)
    pred = model.predict(oof_for_meta)
    return np.sqrt(mean_squared_error(y_for_meta, pred))

def optimize_meta_elastic(trial):
    alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
    model = ElasticNetCV(alphas=[alpha], l1_ratio=[l1_ratio], cv=3)
    model.fit(oof_for_meta, y_for_meta)
    pred = model.predict(oof_for_meta)
    return np.sqrt(mean_squared_error(y_for_meta, pred))

def optimize_meta_mlp(trial):
    hidden_size1 = trial.suggest_int('hidden_size1', 50, 200)
    hidden_size2 = trial.suggest_int('hidden_size2', 20, 100)
    alpha = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
    
    model = EnhancedMLPRegressor(
        hidden_layer_sizes=(hidden_size1, hidden_size2),
        alpha=alpha,
        max_iter=300
    )
    
    # Simple validation
    model.fit(oof_for_meta, y_for_meta)
    pred = model.predict(oof_for_meta)
    return np.sqrt(mean_squared_error(y_for_meta, pred))

print("Optimizing meta-models...")

# Optimize Ridge meta-model
ridge_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
ridge_study.optimize(optimize_meta_ridge, n_trials=50, show_progress_bar=True)
best_ridge_alpha = ridge_study.best_params['alpha']

# Optimize ElasticNet meta-model
elastic_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
elastic_study.optimize(optimize_meta_elastic, n_trials=50, show_progress_bar=True)
best_elastic_params = elastic_study.best_params

# Optimize MLP meta-model
mlp_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
mlp_study.optimize(optimize_meta_mlp, n_trials=50, show_progress_bar=True)
best_mlp_params = mlp_study.best_params

# Train optimized meta-models
meta_models = {
    'ridge_meta': RidgeCV(alphas=[best_ridge_alpha], cv=5),
    'elastic_meta': ElasticNetCV(
        alphas=[best_elastic_params['alpha']], 
        l1_ratio=[best_elastic_params['l1_ratio']],
        cv=5
    ),
    'lasso_meta': LassoCV(alphas=np.logspace(-4, 4, 20), cv=5),
    'mlp_meta': EnhancedMLPRegressor(
        hidden_layer_sizes=(best_mlp_params['hidden_size1'], best_mlp_params['hidden_size2']),
        alpha=best_mlp_params['alpha'],
        max_iter=500
    )
}

best_meta_rmse = float('inf')
best_meta_model = None
best_meta_name = None
meta_results = {}

for meta_name, meta_model in meta_models.items():
    print(f"Training {meta_name}...")
    try:
        if meta_name == 'mlp_meta':
            meta_cv_scores = []
            meta_kfold = KFold(n_splits=3, shuffle=True, random_state=SEED)
            
            for train_idx, val_idx in meta_kfold.split(oof_for_meta):
                X_meta_train, X_meta_val = oof_for_meta[train_idx], oof_for_meta[val_idx]
                y_meta_train, y_meta_val = y_for_meta.iloc[train_idx], y_for_meta.iloc[val_idx]
                
                model_copy = EnhancedMLPRegressor(
                    hidden_layer_sizes=(best_mlp_params['hidden_size1'], best_mlp_params['hidden_size2']),
                    alpha=best_mlp_params['alpha'],
                    max_iter=300
                )
                model_copy.fit(X_meta_train, y_meta_train)
                meta_pred = model_copy.predict(X_meta_val)
                meta_rmse = np.sqrt(mean_squared_error(y_meta_val, meta_pred))
                meta_cv_scores.append(meta_rmse)
            
            meta_rmse = np.mean(meta_cv_scores)
            meta_model.fit(oof_for_meta, y_for_meta)
        else:
            meta_model.fit(oof_for_meta, y_for_meta)
            meta_pred = meta_model.predict(oof_for_meta)
            meta_rmse = np.sqrt(mean_squared_error(y_for_meta, meta_pred))
        
        meta_results[meta_name] = meta_rmse
        print(f"  {meta_name}: {meta_rmse:.4f}")
        
        if meta_rmse < best_meta_rmse:
            best_meta_rmse = meta_rmse
            best_meta_model = meta_model
            best_meta_name = meta_name
            
    except Exception as e:
        print(f"  Error with {meta_name}: {e}")
        meta_results[meta_name] = float('inf')

print(f"\nBest meta-model: {best_meta_name} (RMSE: {best_meta_rmse:.4f})")

# Generate final predictions
final_predictions = best_meta_model.predict(test_predictions)
final_predictions = np.maximum(final_predictions, 0)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

# Save results
filename = f'submission_optimized_meta_model.csv'
submission_df = pd.DataFrame({
    'date': submission_dates,
    'daily_ktCO2': final_predictions
})
submission_df.to_csv(filename, index=False)

print(f"Saved {filename}")
print(f"Prediction stats:")
print(f"  Mean: {final_predictions.mean():.3f}")
print(f"  Std: {final_predictions.std():.3f}")
print(f"  Min: {final_predictions.min():.3f}")
print(f"  Max: {final_predictions.max():.3f}")

# Model performance summary
print(f"\nModel Performance Summary:")
print("-" * 50)
print(f"Best individual model: {min(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[0]} "
      f"(RMSE: {min(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[1]['cv_rmse']:.4f})")
print(f"Best meta-model: {best_meta_name} (RMSE: {best_meta_rmse:.4f})")

# Save detailed results
results_df = pd.DataFrame([
    {'model': name, 'cv_rmse': scores['cv_rmse'], 'cv_std': scores['cv_std'], 
     'selected': name in selected_model_names}
    for name, scores in model_scores.items()
]).sort_values('cv_rmse')

results_df.to_csv('model_performance_detailed.csv', index=False)
print(f"Detailed results saved to 'model_performance_detailed.csv'")

print("\nOptimized hyperparameters:")
for model_name, params in best_params.items():
    print(f"{model_name}: {params}")

print(f"\nMeta-model optimization results:")
for meta_name, rmse in meta_results.items():
    print(f"{meta_name}: {rmse:.4f}")























# TENSORFLOW SEQUENTIAL MODELS (OPTIONAL) + Note
# Uncomment this section if you want to use TensorFlow

"""
# TensorFlow Sequential Models (Optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Set TensorFlow random seed
    tf.random.set_seed(SEED)
    
    class TensorFlowSequentialModel:
        def __init__(self, seq_length=14, n_features=None, units=64):
            self.seq_length = seq_length
            self.n_features = n_features
            self.units = units
            self.model = None
            self.scaler = StandardScaler()
            
        def build_model(self, n_features):
            model = keras.Sequential([
                layers.LSTM(self.units, return_sequences=True, input_shape=(self.seq_length, n_features)),
                layers.Dropout(0.2),
                layers.LSTM(self.units//2, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            return model
        
        def create_sequences(self, X, y=None):
            if len(X) <= self.seq_length:
                return None, None
                
            sequences = []
            targets = [] if y is not None else None
            
            X_scaled = self.scaler.fit_transform(X) if not hasattr(self, '_fitted') else self.scaler.transform(X)
            self._fitted = True
            
            for i in range(self.seq_length, len(X_scaled)):
                sequences.append(X_scaled[i-self.seq_length:i])
                if y is not None:
                    targets.append(y.iloc[i] if hasattr(y, 'iloc') else y[i])
            
            return np.array(sequences), np.array(targets) if targets is not None else None
        
        def fit(self, X, y):
            X_seq, y_seq = self.create_sequences(X, y)
            if X_seq is None:
                raise ValueError("Not enough data for sequence creation")
                
            self.n_features = X_seq.shape[2]
            self.model = self.build_model(self.n_features)
            
            # Train with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            self.model.fit(
                X_seq, y_seq,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            return self
        
        def predict(self, X):
            X_seq, _ = self.create_sequences(X, None)
            if X_seq is None:
                return np.full(len(X), np.mean(y_for_meta))
                
            pred_seq = self.model.predict(X_seq, verbose=0).flatten()
            
            # Pad predictions for initial sequence
            full_pred = np.full(len(X), pred_seq[0] if len(pred_seq) > 0 else 0)
            full_pred[self.seq_length:] = pred_seq
            return full_pred
    
    # Add TensorFlow models to the model list
    tf_models = {
        'tensorflow_lstm': TensorFlowSequentialModel(seq_length=14, units=64),
        'tensorflow_lstm_deep': TensorFlowSequentialModel(seq_length=21, units=128),
    }
    
    print("TensorFlow models are available!")
    print("Add these to your models dictionary to use them:")
    for name in tf_models.keys():
        print(f"  - {name}")
        
except ImportError:
    print("TensorFlow not installed. Using sklearn-based sequential models instead.")
    print("To use TensorFlow models, install: pip install tensorflow==2.13.0")
"""


"""
param
Best lightgbm RMSE: 1.7049
Best params: {'learning_rate': 0.05127862283124463, 'num_leaves': 245, 'max_depth': 10, 'min_child_samples': 23, 'subsample': 0.6176714838282072, 'colsample_bytree': 0.878721519175876, 'reg_alpha': 9.923772567588348e-06, 'reg_lambda': 2.76835469353685}

Best xgboost RMSE: 1.6238
Best params: {'learning_rate': 0.08204180423731024, 'max_depth': 7, 'subsample': 0.920817137596488, 'colsample_bytree': 0.8160571823010057, 'reg_alpha': 3.561302339114832e-07, 'reg_lambda': 2.3257577031009327e-05}

Best catboost RMSE: 1.5590
Best params: {'learning_rate': 0.0585971302788046, 'depth': 10, 'l2_leaf_reg': 0.010842262717330166, 'iterations': 1163, 'border_count': 123, 'bagging_temperature': 0.7600340105889054}

Best random_forest RMSE: 1.9617
Best params: {'n_estimators': 340, 'max_depth': 22, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None}

Best gradient_boosting RMSE: 1.7262
Best params: {'n_estimators': 461, 'learning_rate': 0.018254861693279043, 'max_depth': 7, 'subsample': 0.8631663038344568}

Best extra_trees RMSE: 1.5225
Best params: {'n_estimators': 420, 'max_depth': 22, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_features': None}
"""
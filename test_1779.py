import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from tqdm.notebook import tqdm

print("Step 1: Loading and Cleaning Data...")

train_df = pd.read_csv('datatrain.csv')
test_df = pd.read_csv('datatest.csv')
submission_dates = test_df['date']
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

train_ids = train_df['obs_id']
test_ids = test_df['obs_id']
train_y = train_df['daily_ktCO2']

print(f"Target stats - Mean: {train_y.mean():.3f}, Std: {train_y.std():.3f}")
print(f"Target range: [{train_y.min():.3f}, {train_y.max():.3f}]")

train_df = train_df.drop(columns=['daily_ktCO2', 'obs_id'])
test_df = test_df.drop(columns=['obs_id'])

combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
print(combined_df.info())
print(combined_df)

text_cols = combined_df.select_dtypes(include=['object']).columns
for col in text_cols:
    if col != 'date':
        combined_df[col] = combined_df[col].astype(str).str.lower().str.strip()

bool_cols = ['rain', 'weekend', 'holiday']
for col in bool_cols:
    if col in combined_df.columns:
        map_dict = {'true': "yes", 'yes': "yes", '1': "yes", 'false': "no", 'no': "no", '0': "no"}
        combined_df[col] = combined_df[col].astype(str).str.strip().str.lower().map(map_dict)

for col in bool_cols:
    print(combined_df[col].unique())

combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['so2'] = pd.to_numeric(combined_df['so2'], errors='coerce')
numerical_cols = [col for col in combined_df.columns if combined_df[col].dtype in [np.int64, np.float64]]
categorical_cols = [col for col in combined_df.columns if combined_df[col].dtype == 'object']

missing_analysis = pd.DataFrame({
    'column': combined_df.columns,
    'missing_count': combined_df.isnull().sum(),
    'missing_pct': (combined_df.isnull().sum() / len(combined_df)) * 100
})
missing_analysis = missing_analysis[missing_analysis['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("Columns with missing values:")
print(missing_analysis)

high_missing_cols = missing_analysis[missing_analysis['missing_pct'] > 70]['column'].tolist()
if high_missing_cols:
    combined_df = combined_df.drop(columns=high_missing_cols)
    print(f"\nDropped {len(high_missing_cols)} columns with >70% missing data:")
    print(high_missing_cols)

combined_df["doy"] = combined_df["date"].dt.dayofyear
no2_seasonal_avg = combined_df.groupby("doy")["no2"].mean()
combined_df["no2_imputed"] = combined_df.apply(
    lambda row: no2_seasonal_avg[row["doy"]] if pd.isna(row["no2"]) else row["no2"],
    axis=1
)

o3_seasonal_avg = combined_df.groupby("doy")["o3"].mean()
combined_df["o3_imputed"] = combined_df.apply(
    lambda row: o3_seasonal_avg[row["doy"]] if pd.isna(row["o3"]) else row["o3"],
    axis=1
)

tci_seasonal_avg = combined_df.groupby("doy")["TCI"].mean()
combined_df["TCI_imputed"] = combined_df.apply(
    lambda row: tci_seasonal_avg[row["doy"]] if pd.isna(row["TCI"]) else row["TCI"],
    axis=1
)

so2_seasonal_avg = combined_df.groupby("doy")["so2"].mean()
combined_df["so2_imputed"] = combined_df.apply(
    lambda row: so2_seasonal_avg[row["doy"]] if pd.isna(row["so2"]) else row["so2"],
    axis=1
)

combined_df['so2_imputed'] = combined_df['so2_imputed'].interpolate()

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

cyclical_to_drop = ['month', 'day_of_week', 'day_of_year', 'week_of_year']
combined_df = combined_df.drop(columns=[col for col in cyclical_to_drop if col in combined_df.columns])

weather_cols = ['temp', 'tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure']
pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2']

if all(col in combined_df.columns for col in ['temp', 'tempmax', 'tempmin']):
    combined_df['temp_range'] = combined_df['tempmax'] - combined_df['tempmin']
    combined_df['temp_variance'] = combined_df['temp_range'] / (combined_df['temp'].replace(0, 1e-6) + 1e-6) 

if 'temp' in combined_df.columns and 'humidity' in combined_df.columns:
    combined_df['heat_index'] = combined_df['temp'] * (1 + combined_df['humidity'] / 100)
    combined_df['comfort_index'] = combined_df['temp'] * (1 - abs(combined_df['humidity'] - 50) / 100)

wind_features = ['windspeed', 'windgust', 'windspeedmax', 'windspeedmin']
for feat in wind_features:
    if feat in combined_df.columns and 'temp' in combined_df.columns:
        combined_df[f'{feat}_temp_interaction'] = combined_df[feat] * combined_df['temp']

if 'sealevelpressure' in combined_df.columns:
    combined_df['pressure_anomaly'] = combined_df['sealevelpressure'] - combined_df['sealevelpressure'].mean()

if len([col for col in pollutant_cols if col in combined_df.columns]) > 1:
    available_pollutants = [col for col in pollutant_cols if col in combined_df.columns]
    combined_df['pollutant_sum'] = combined_df[available_pollutants].sum(axis=1)
    combined_df['pollutant_mean'] = combined_df[available_pollutants].mean(axis=1)
    
    if 'pm25' in combined_df.columns and 'pm10' in combined_df.columns:
        combined_df['fine_coarse_ratio'] = combined_df['pm25'] / (combined_df['pm10'] + 1e-6)
    
    if 'no2' in combined_df.columns and 'o3' in combined_df.columns:
        combined_df['no2_o3_ratio'] = combined_df['no2_imputed'] / (combined_df['o3_imputed'] + 1e-6)

solar_features = ['solarradiation', 'solarenergy', 'uvindex']
for feat in solar_features:
    if feat in combined_df.columns and 'cloudcover' in combined_df.columns:
        combined_df[f'{feat}_cloud_interaction'] = combined_df[feat] * (100 - combined_df['cloudcover']) / 100

for col in categorical_cols:
    combined_df[col] = combined_df[col].replace("nan", np.nan)
    if combined_df[col].isnull().any():
        mode_vals = combined_df[col].mode()
        fill_val = mode_vals[0] if len(mode_vals) > 0 else 'unknown'
        combined_df[col] = combined_df[col].fillna(fill_val)

combined_df = combined_df.drop(['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'week_of_year_sin', 'week_of_year_cos'], axis=1)

combined_df['pm10_log'] = np.log1p(combined_df['pm10'])
combined_df = combined_df.drop(['pm10'], axis=1)

oh_encode_cols = ['rain', 'conditions', 'seasons', 'weekend', 'holiday']
combined_df = pd.get_dummies(combined_df, columns=oh_encode_cols, drop_first=True)

combined_df['pm_ratio'] = combined_df['pm25'] / (combined_df['pm10_log'])

numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    corr_matrix = combined_df[numerical_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
    print(high_corr_features)

bool_cols = combined_df.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    combined_df[col] = combined_df[col].astype(int)

combined_df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in combined_df.columns]
combined_df.columns = [re.sub(r'_+', '_', col).strip('_') for col in combined_df.columns]

combined_df = combined_df.sort_values('date')

features_to_lag = ['pm25', 'temp', 'humidity', 'windspeed', 'o3_imputed', 'no2_imputed', 'pollutant_sum']
lag_periods = [1, 2, 3, 7, 14]
rolling_windows = [3, 7, 14]

for col in features_to_lag:
    for lag in lag_periods:
        combined_df[f'{col}_lag_{lag}'] = combined_df[col].shift(lag)
    
    for window in rolling_windows:
        combined_df[f'{col}_roll_mean_{window}'] = combined_df[col].shift(1).rolling(window=window).mean()
        combined_df[f'{col}_roll_std_{window}'] = combined_df[col].shift(1).rolling(window=window).std()

combined_df.fillna(method='ffill', inplace=True)
for col in combined_df.select_dtypes(include=[np.number]).columns:
    if combined_df[col].isna().any():
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

combined_df = combined_df.drop(['o3', 'no2', 'so2', 'TCI'], axis=1)
combined_df.info()

X_train = combined_df.iloc[train_ids-1, :].drop(['date'], axis=1)
X_test = combined_df.iloc[test_ids-1, :].drop(['date'], axis=1)

class AdvancedMLPRegressor:
    def __init__(self, hidden_layer_sizes=(200, 100, 50), alpha=0.01, max_iter=500):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate='adaptive',
            solver='adam',
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def get_models():
    best_params = {
        "lightgbm": {
            "learning_rate": 0.08792390029299216,
            "num_leaves": 148,
            "max_depth": 18,
            "min_child_samples": 100,
            "subsample": 0.8880829759865683,
            "colsample_bytree": 0.6770025154580377,
            "bagging_freq": 2,
            "bagging_fraction": 0.8472562306706372,
            "feature_fraction": 0.8913366503066713,
            "reg_lambda":0.30328406350123954,
            "reg_alpha": 0.001362235909930096
        },
        "xgboost": {
            "learning_rate": 0.019888031461512364,
            "max_depth": 4,
            "subsample": 0.6766099811158373,
            "colsample_bytree": 0.9911186210777034,
            "reg_lambda": 7.537856205877511,
            "reg_alpha": 0.002026025671840752
        },
        "catboost": {
            "learning_rate": 0.04398925741891746,
            "depth": 6,
            "l2_leaf_reg": 0.00023958385724389604
        },
        "random_forest": {
            "n_estimators": 218,
            "max_depth": 16,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        },
        "extra_trees": {'n_estimators': 250, 'max_depth': 14, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        'gradient_boosting': {'n_estimators': 328, 'learning_rate': 0.08541807286319114, 'max_depth': 4}
    }
    
    models = {
        'lightgbm': lgb.LGBMRegressor(
            **best_params['lightgbm'],
            random_state=42, n_estimators=1000, verbose=-1
        ),
        'xgboost': xgb.XGBRegressor(
            **best_params['xgboost'],
            random_state=42, n_estimators=1000, verbosity=0
        ),
        'catboost': cb.CatBoostRegressor(
            **best_params['catboost'],
            random_state=42, iterations=1000, verbose=0
        ),
        'random_forest': RandomForestRegressor(
            **best_params['random_forest'],
            random_state=42, n_jobs=-1
        ),
        'extra_trees': ExtraTreesRegressor(
            **best_params['extra_trees'],
            random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            **best_params['gradient_boosting'],
            random_state=42
        ),
        'ridge': RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5),
        'elastic_net': ElasticNetCV(alphas=np.logspace(-4, 4, 10), 
                                   l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5),
        'lasso': LassoCV(alphas=np.logspace(-4, 4, 20), cv=5),
    }
    
    return models

# Cross-validation setup
NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Single model evaluation
print("\n" + "="*60)
print("INDIVIDUAL MODEL PERFORMANCE (INCLUDING SEQUENTIAL MODELS)")
print("="*60)

models = get_models()
model_scores = {}
model_predictions = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    try:
        cv_scores = cross_val_score(model, X_train, train_y, cv=kfold, 
                                   scoring='neg_root_mean_squared_error', n_jobs=1 if 'sequential' in name else -1)
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

print(f"\n{'Model':<18} {'CV RMSE':<10} {'Std':<10}")
print("-" * 40)
for name, scores in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse']):
    print(f"{name:<18} {scores['cv_rmse']:<10.4f} {scores['cv_std']:<10.4f}")

print("\n" + "="*60)
print("ENSEMBLE METHODS WITH SEQUENTIAL MODELS")
print("="*60)

selected_model_names = [name for name, _ in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[:7]]
print(f"Selected top 7 models: {selected_model_names}")

oof_predictions = np.zeros((len(X_train), len(selected_model_names)))
test_predictions = np.zeros((len(X_test), len(selected_model_names)))

models_for_stacking = get_models()

for i, name in enumerate(selected_model_names):
    print(f"--- Stacking training for {name} ---")
    model = models_for_stacking[name]
    fold_test_preds = []
    
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

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_ids], X_train.iloc[val_ids]
        y_fold_train, y_fold_val = train_y.iloc[train_ids], train_y.iloc[val_ids]
        
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

meta_models = {
    'ridge_meta': RidgeCV(alphas=np.logspace(-4, 4, 20)),
    'elastic_meta': ElasticNetCV(alphas=np.logspace(-4, 4, 10), 
                                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]),
    'lasso_meta': LassoCV(alphas=np.logspace(-4, 4, 20)),
    'mlp_meta': AdvancedMLPRegressor(
        hidden_layer_sizes=(100, 50, 25), 
        alpha=0.01,
        max_iter=500
    )
}

best_meta_rmse = float('inf')
best_meta_model = None
best_meta_name = None
best_meta_std = 0
meta_results = {}

for meta_name, meta_model in meta_models.items():
    print(f"Training {meta_name}...")
    try:
        if meta_name == 'mlp_meta':
            meta_cv_scores = []
            meta_kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            
            for train_idx, val_idx in meta_kfold.split(oof_for_meta):
                X_meta_train, X_meta_val = oof_for_meta[train_idx], oof_for_meta[val_idx]
                y_meta_train, y_meta_val = y_for_meta.iloc[train_idx], y_for_meta.iloc[val_idx]
                
                model_copy = AdvancedMLPRegressor(
                    hidden_layer_sizes=(100, 50, 25), 
                    alpha=0.01,
                    max_iter=300
                )
                model_copy.fit(X_meta_train, y_meta_train)
                meta_pred = model_copy.predict(X_meta_val)
                meta_rmse = np.sqrt(mean_squared_error(y_meta_val, meta_pred))
                meta_cv_scores.append(meta_rmse)
            
            meta_rmse = np.mean(meta_cv_scores)
            meta_std = np.std(meta_cv_scores)
            print(f"  {meta_name}: {meta_rmse:.4f} (±{meta_std:.4f})")
            meta_model.fit(oof_for_meta, y_for_meta)
        else:
            meta_model.fit(oof_for_meta, y_for_meta)
            meta_pred = meta_model.predict(oof_for_meta)
            meta_rmse = np.sqrt(mean_squared_error(y_for_meta, meta_pred))
            meta_std = 0
            print(f"  {meta_name}: {meta_rmse:.4f}")
        
        meta_results[meta_name] = meta_rmse
        if meta_rmse < best_meta_rmse:
            best_meta_rmse = meta_rmse
            best_meta_std = meta_std
            best_meta_model = meta_model
            best_meta_name = meta_name
            
    except Exception as e:
        print(f"  Error with {meta_name}: {e}")
        meta_results[meta_name] = float('inf')

print(f"\nBest meta-model: {best_meta_name} (RMSE: {best_meta_rmse:.4f})")

final_predictions = best_meta_model.predict(test_predictions)
final_predictions = np.maximum(final_predictions, 0)

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

filename = f'submission_meta_model.csv'
submission_df = pd.DataFrame({
    'date': submission_dates,
    'daily_ktCO2': final_predictions
})
submission_df.to_csv(filename, index=False)
print(f"Saved {filename}")
print(f"  Mean: {final_predictions.mean():.3f}, Std: {final_predictions.std():.3f}, Min: {final_predictions.min():.3f}, Max: {final_predictions.max():.3f}")

print(f"\nModel performance summary:")
print(f"Best individual model: {min(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[0]} "
      f"(RMSE: {min(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[1]['cv_rmse']:.4f})")
print(f"Best meta-model: {best_meta_name} (RMSE: {best_meta_rmse:.4f})")
# CO2 Prediction - Enhanced Machine Learning Pipeline

## Import Libraries and Setup

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler

import holidays
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

# =============================================================================
# 1. DATA LOADING AND BASIC CLEANING
# =============================================================================

def load_and_prepare_data():
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
    
    return combined_df, train_ids, test_ids, train_y, submission_dates

def clean_text_and_boolean_columns(df):
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        if col != 'date':
            df[col] = df[col].astype(str).str.lower().str.strip()
    
    bool_cols = ['rain', 'weekend', 'holiday']
    for col in bool_cols:
        if col in df.columns:
            map_dict = {'true': "yes", 'yes': "yes", '1': "yes", 
                       'false': "no", 'no': "no", '0': "no"}
            df[col] = df[col].astype(str).str.strip().str.lower().map(map_dict)
    
    df['date'] = pd.to_datetime(df['date'])
    df['so2'] = pd.to_numeric(df['so2'], errors='coerce')
    
    return df

# =============================================================================
# 2. MISSING DATA ANALYSIS AND HANDLING
# =============================================================================

def analyze_missing_data(df):
    missing_analysis = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100
    })
    missing_analysis = missing_analysis[missing_analysis['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    print("Columns with missing values:")
    print(missing_analysis)
    return missing_analysis

def drop_high_missing_columns(df, threshold=70):
    missing_analysis = analyze_missing_data(df)
    high_missing_cols = missing_analysis[missing_analysis['missing_pct'] > threshold]['column'].tolist()
    
    if high_missing_cols:
        df = df.drop(columns=high_missing_cols)
        print(f"\nDropped {len(high_missing_cols)} columns with >{threshold}% missing data:")
        print(high_missing_cols)
    
    return df

def impute_by_seasonal_average(df, columns):
    df["doy"] = df["date"].dt.dayofyear
    
    for col in columns:
        if col in df.columns:
            seasonal_avg = df.groupby("doy")[col].mean()
            df[f"{col}_imputed"] = df.apply(
                lambda row: seasonal_avg[row["doy"]] if pd.isna(row[col]) else row[col],
                axis=1
            )
    
    df = df.drop(['doy'], axis=1)
    return df

def interpolate_missing_values(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].interpolate()
    return df

def fill_categorical_missing(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'date':
            # Convert literal "nan" strings to np.nan
            df[col] = df[col].replace("nan", np.nan)
            
            # Fill with mode or 'unknown'
            if df[col].isnull().any():
                mode_vals = df[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else 'unknown'
                df[col] = df[col].fillna(fill_val)
    
    return df

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def add_time_features(df):
    """Add comprehensive time-based features"""
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    df['weekend'] = df['day_of_week'].apply(lambda x: 'yes' if x >= 5 else 'no')
    
    indo_holidays = holidays.CountryHoliday('ID')
    df['holiday'] = df['date'].isin(indo_holidays).map({True: 'yes', False: 'no'})
    
    return df

def add_cyclical_features(df, col, period):
    if col in df.columns:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df

def create_cyclical_encodings(df):
    if 'winddir' in df.columns:
        df = add_cyclical_features(df, 'winddir', 360)
        df = df.drop('winddir', axis=1)
    
    df = add_cyclical_features(df, 'month', 12)
    df = add_cyclical_features(df, 'day_of_week', 7)
    df = add_cyclical_features(df, 'day_of_year', 365)
    df = add_cyclical_features(df, 'week_of_year', 52)
    df = add_cyclical_features(df, 'quarter', 4)
    
    cyclical_to_drop = ['month', 'day_of_week', 'day_of_year', 'week_of_year']
    df = df.drop(columns=[col for col in cyclical_to_drop if col in df.columns])
    
    return df

def create_weather_interactions(df):
    if all(col in df.columns for col in ['temp', 'tempmax', 'tempmin']):
        df['temp_range'] = df['tempmax'] - df['tempmin']
        df['temp_variance'] = df['temp_range'] / (df['temp'].replace(0, 1e-6) + 1e-6)
    
    if 'temp' in df.columns and 'humidity' in df.columns:
        df['heat_index'] = df['temp'] * (1 + df['humidity'] / 100)
        df['comfort_index'] = df['temp'] * (1 - abs(df['humidity'] - 50) / 100)
    
    wind_features = ['windspeed', 'windgust', 'windspeedmax', 'windspeedmin']
    for feat in wind_features:
        if feat in df.columns and 'temp' in df.columns:
            df[f'{feat}_temp_interaction'] = df[feat] * df['temp']
    
    if 'sealevelpressure' in df.columns:
        df['pressure_anomaly'] = df['sealevelpressure'] - df['sealevelpressure'].mean()
    
    return df

def create_pollutant_features(df):
    pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2']
    available_pollutants = [col for col in pollutant_cols if col in df.columns]
    
    if len(available_pollutants) > 1:
        df['pollutant_sum'] = df[available_pollutants].sum(axis=1)
        df['pollutant_mean'] = df[available_pollutants].mean(axis=1)
        
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['fine_coarse_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)
        
        if 'no2_imputed' in df.columns and 'o3_imputed' in df.columns:
            df['no2_o3_ratio'] = df['no2_imputed'] / (df['o3_imputed'] + 1e-6)
    
    return df

def create_solar_interactions(df):
    solar_features = ['solarradiation', 'solarenergy', 'uvindex']
    for feat in solar_features:
        if feat in df.columns and 'cloudcover' in df.columns:
            df[f'{feat}_cloud_interaction'] = df[feat] * (100 - df['cloudcover']) / 100
    
    return df

def create_lag_and_rolling_features(df, features_to_lag, lag_periods, rolling_windows):
    df = df.sort_values('date')
    
    for col in features_to_lag:
        if col in df.columns:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            for window in rolling_windows:
                df[f'{col}_roll_mean_{window}'] = df[col].shift(1).rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].shift(1).rolling(window=window).std()
    
    return df

# =============================================================================
# 4. ADVANCED ML MODELS
# =============================================================================

class AdvancedMLPRegressor:
    """Enhanced MLP with better hyperparameters"""
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

class AdvancedSVR:
    """SVR with RBF kernel and good hyperparameters"""
    def __init__(self, C=100, gamma='scale', epsilon=0.1):
        self.model = SVR(C=C, gamma=gamma, epsilon=epsilon, kernel='rbf')
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def get_optimized_models():
    """Return dictionary of models with optimized hyperparameters"""
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
            "reg_lambda": 0.30328406350123954,
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
        "extra_trees": {
            'n_estimators': 250, 
            'max_depth': 14, 
            'min_samples_split': 5, 
            'min_samples_leaf': 1, 
            'max_features': 'sqrt'
        },
        'gradient_boosting': {
            'n_estimators': 328, 
            'learning_rate': 0.08541807286319114, 
            'max_depth': 4
        }
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
        'elastic_net': ElasticNetCV(
            alphas=np.logspace(-4, 4, 10), 
            l1_ratio=[0.1, 0.5, 0.7, 0.9], 
            cv=5
        ),
        'lasso': LassoCV(alphas=np.logspace(-4, 4, 20), cv=5),
    }
    
    return models

# =============================================================================
# 5. MAIN EXECUTION PIPELINE
# =============================================================================
def main_pipeline():
    combined_df, train_ids, test_ids, train_y, submission_dates = load_and_prepare_data()
    
    combined_df = clean_text_and_boolean_columns(combined_df)
    
    combined_df = drop_high_missing_columns(combined_df, threshold=70)
    
    seasonal_impute_cols = ['no2', 'o3', 'TCI', 'so2']
    combined_df = impute_by_seasonal_average(combined_df, seasonal_impute_cols)
    
    interpolate_cols = ['so2_imputed']
    combined_df = interpolate_missing_values(combined_df, interpolate_cols)
    
    combined_df = fill_categorical_missing(combined_df)
    
    combined_df = add_time_features(combined_df)
    combined_df = create_cyclical_encodings(combined_df)
    combined_df = create_weather_interactions(combined_df)
    combined_df = create_pollutant_features(combined_df)
    combined_df = create_solar_interactions(combined_df)
    
    to_remove = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 
                 'week_of_year_sin', 'week_of_year_cos']
    combined_df = combined_df.drop([col for col in to_remove if col in combined_df.columns], axis=1)
    
    if 'pm10' in combined_df.columns:
        combined_df['pm10_log'] = np.log1p(combined_df['pm10'])
        combined_df = combined_df.drop(['pm10'], axis=1)
    
    oh_encode_cols = ['rain', 'conditions', 'seasons', 'weekend', 'holiday']
    oh_encode_cols = [col for col in oh_encode_cols if col in combined_df.columns]
    combined_df = pd.get_dummies(combined_df, columns=oh_encode_cols, drop_first=True)
    
    if 'pm25' in combined_df.columns and 'pm10_log' in combined_df.columns:
        combined_df['pm_ratio'] = combined_df['pm25'] / combined_df['pm10_log']
    
    bool_cols = combined_df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        combined_df[col] = combined_df[col].astype(int)
    
    combined_df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in combined_df.columns]
    combined_df.columns = [re.sub(r'_+', '_', col).strip('_') for col in combined_df.columns]
    
    combined_df = combined_df.sort_values('date')
    
    features_to_lag = ['pm25', 'temp', 'humidity', 'windspeed', 'o3_imputed', 'no2_imputed', 'pollutant_sum']
    lag_periods = [1, 2, 3, 7, 14]
    rolling_windows = [3, 7, 14]
    
    combined_df = create_lag_and_rolling_features(combined_df, features_to_lag, lag_periods, rolling_windows)
    
    combined_df = combined_df.fillna(method='ffill')
    for col in combined_df.select_dtypes(include=[np.number]).columns:
        if combined_df[col].isna().any():
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())
    
    cols_to_remove = ['o3', 'no2', 'so2', 'TCI']
    combined_df = combined_df.drop([col for col in cols_to_remove if col in combined_df.columns], axis=1)
    
    print(f"Final feature set shape: {combined_df.shape}")
    print(f"Final feature set info:")
    print(combined_df.info())
    
    X_train = combined_df.iloc[train_ids-1, :].drop(['date'], axis=1)
    X_test = combined_df.iloc[test_ids-1, :].drop(['date'], axis=1)
    
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    NFOLDS = 5
    kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)
    
    models = get_optimized_models()
    model_scores = {}
    model_predictions = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        try:
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
    
    print(f"\n{'Model':<18} {'CV RMSE':<10} {'Std':<10}")
    print("-" * 40)
    for name, scores in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse']):
        print(f"{name:<18} {scores['cv_rmse']:<10.4f} {scores['cv_std']:<10.4f}")
    
    print("\n" + "="*60)
    print("ENSEMBLE METHODS WITH STACKING")
    print("="*60)
    
    selected_model_names = [name for name, _ in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[:7]]
    print(f"Selected top 7 models: {selected_model_names}")
    
    oof_predictions = np.zeros((len(X_train), len(selected_model_names)))
    test_predictions = np.zeros((len(X_test), len(selected_model_names)))
    
    models_for_stacking = get_optimized_models()
    
    for i, name in enumerate(selected_model_names):
        print(f"--- Stacking training for {name} ---")
        model = models_for_stacking[name]
        fold_test_preds = []
        
        for fold, (train_ids_fold, val_ids) in enumerate(kfold.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_ids_fold], X_train.iloc[val_ids]
            y_fold_train, y_fold_val = train_y.iloc[train_ids_fold], train_y.iloc[val_ids]
            
            try:
                model.fit(X_fold_train, y_fold_train)
                val_pred = model.predict(X_fold_val)
                oof_predictions[val_ids, i] = val_pred
            except Exception as e:
                print(f"    Fold {fold} error: {e}")
                oof_predictions[val_ids, i] = np.mean(y_fold_train)
        
        for fold, (train_ids_fold, val_ids) in enumerate(kfold.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_ids_fold], X_train.iloc[val_ids]
            y_fold_train, y_fold_val = train_y.iloc[train_ids_fold], train_y.iloc[val_ids]
            
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
    
    for meta_name, meta_model in meta_models.items():
        print(f"Training {meta_name}...")
        
        try:
            if meta_name == 'mlp_meta':
                # Use cross-validation for MLP
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
                    meta_rmse = np.sqrt(mean_squared_error(y_for_meta, meta_pred))
            
            print(f"  {meta_name}: {meta_rmse:.4f}")
            
            if meta_rmse < best_meta_rmse:
                best_meta_rmse = meta_rmse
                best_meta_model = meta_model
                best_meta_name = meta_name
                
        except Exception as e:
            print(f"  Error with {meta_name}: {e}")
    
    print(f"\nBest meta-model: {best_meta_name} (RMSE: {best_meta_rmse:.4f})")
    
    final_predictions = best_meta_model.predict(test_predictions)
    final_predictions = np.maximum(final_predictions, 0)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    filename = 'submission_enhanced.csv'
    submission_df = pd.DataFrame({
        'date': submission_dates,
        'daily_ktCO2': final_predictions
    })
    submission_df.to_csv(filename, index=False)
    
    print(f"Saved {filename}")
    print(f"  Mean: {final_predictions.mean():.3f}, Std: {final_predictions.std():.3f}")
    print(f"  Min: {final_predictions.min():.3f}, Max: {final_predictions.max():.3f}")
    
    print(f"\nModel performance summary:")
    best_individual = min(model_scores.items(), key=lambda x: x[1]['cv_rmse'])
    print(f"Best individual model: {best_individual[0]} (RMSE: {best_individual[1]['cv_rmse']:.4f})")
    print(f"Best meta-model: {best_meta_name} (RMSE: {best_meta_rmse:.4f})")
    
    return submission_df, model_scores, best_meta_rmse

# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================
def plot_feature_importance(model, feature_names, title, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
        scores_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=scores_df.head(top_n), x='importance', y='feature')
        plt.title(f'{title} - Top {top_n} Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return scores_df
    else:
        print(f"Model {type(model).__name__} doesn't have feature_importances_ attribute")
        return None

def analyze_predictions(y_true, y_pred, title="Prediction Analysis"):
    from sklearn.metrics import mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{title}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} - Scatter Plot')
    
    plt.subplot(1, 2, 2)
    residuals = y_pred - y_true
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{title} - Residuals')
    
    plt.tight_layout()
    plt.show()
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def display_data_summary(df):
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print(f"\nNo missing values found!")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\nNumerical features: {len(numerical_cols)}")
        print(df[numerical_cols].describe())
    
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical features: {len(categorical_cols)}")
        for col in categorical_cols:
            print(f"{col}: {df[col].nunique()} unique values")

# =============================================================================
# 7. EXECUTE MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CO2 EMISSION PREDICTION - ENHANCED ML PIPELINE")
    print("="*80)
    
    submission_df, model_scores, best_meta_rmse = main_pipeline()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Final submission saved with {len(submission_df)} predictions")
    print(f"Best meta-model RMSE: {best_meta_rmse:.4f}")
    
    print("\nFirst 10 predictions:")
    print(submission_df.head(10))

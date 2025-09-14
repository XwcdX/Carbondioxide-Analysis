import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
import re
#ini mek untuk biar nek error pendek
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. Load and Clean Data
# -----------------------------------------------------------------
print("Step 1: Loading and Cleaning Data...")
try:
    train_df = pd.read_csv('datatrain.csv')
    test_df = pd.read_csv('datatest.csv')
    submission_dates = test_df['date']
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
except FileNotFoundError:
    print("Make sure 'datatrain.csv' and 'datatest.csv' are in the same directory.")
    exit()

# simpen id data sama result CO2
train_ids = train_df['obs_id']
test_ids = test_df['obs_id']
train_y = train_df['daily_ktCO2']

print(f"Target stats - Mean: {train_y.mean():.3f}, Std: {train_y.std():.3f}")
print(f"Target range: [{train_y.min():.3f}, {train_y.max():.3f}]")

# buang co2 sm id datae karena biar datae lebih banyak sekalian :v
train_df = train_df.drop(columns=['daily_ktCO2', 'obs_id'])
test_df = test_df.drop(columns=['obs_id'])

# gabung kedua data
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# bersihin sng capslock"
text_cols = combined_df.select_dtypes(include=['object']).columns
for col in text_cols:
    if col != 'date':
        combined_df[col] = combined_df[col].astype(str).str.lower().str.strip()

# ini sng true false
bool_cols = ['rain', 'weekend', 'holiday']
for col in bool_cols:
    if col in combined_df.columns:
        map_dict = {'true': True, 'yes': True, '1': True, 'false': False, 'no': False, '0': False}
        combined_df[col] = combined_df[col].astype(str).str.strip().str.lower().map(map_dict)
        combined_df[col] = combined_df[col].fillna(False)

print("Data cleaning complete.")

# -----------------------------------------------------------------
# 2. Enhanced Feature Engineering
# -----------------------------------------------------------------
print("\nStep 2: Feature Engineering...")
# semua fitur untuk tanggal -> nambah fitur
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

combined_df = combined_df.drop('date', axis=1)

# ini yang sin cos untuk winddir sama sekalian untuk sng bulan hari tahun 
def add_cyclical_features(df, col, period):
    """Add sine/cosine encoding for cyclical features"""
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

# buang fitur" yang udah di cyclical
cyclical_to_drop = ['month', 'day_of_week', 'day_of_year']
combined_df = combined_df.drop(columns=[col for col in cyclical_to_drop if col in combined_df.columns])

# Feature interactions -> gabungin fitur" yang kemungkinan kalau dicombine bagus
# ini contoh e nek suhu panas tapi ga lembab biasae ttp ga keringetan or keringet e cepet nguap sebalik e ya lembab jd ttp panas
if 'temp' in combined_df.columns and 'humidity' in combined_df.columns:
    combined_df['temp_humidity'] = combined_df['temp'] * combined_df['humidity']

# ini biar tau temperatur di macem" windspeed_temp atau biar ada kek perasaan temperatur yang dirasakan
# contoh nek angin e kenceng 30C serasa lebih dingin dibanding nek angin e pelan ws panas gada angin lagi :v
if 'windspeed' in combined_df.columns and 'temp' in combined_df.columns:
    combined_df['windspeed_temp'] = combined_df['windspeed'] * combined_df['temp']

# ini biar model paham tentang partikel" 2.5 itu biasae dari manusia kek asap kendaraan
# 10 itu biasae alami kek pollen 
# tujuan e dibagi jadi makin banyak sng pm 10 model jd tau posisinya lagi banyak sng alami
# bisa dibilang semisal pm_ratio e kecil atau pm10 e besar maka polusie alami bukan buatan dan sebalik e
if 'pm25' in combined_df.columns and 'pm10' in combined_df.columns:
    combined_df['pm_ratio'] = combined_df['pm25'] / (combined_df['pm10'] + 1e-6)
print("Feature engineering complete.")

# -----------------------------------------------------------------
# 3. Smart Missing Value Handling
# -----------------------------------------------------------------
print("\nStep 3: Handling Missing Values...")
missing_analysis = pd.DataFrame({
    'column': combined_df.columns,
    'missing_count': combined_df.isnull().sum(),
    'missing_pct': (combined_df.isnull().sum() / len(combined_df)) * 100
})
missing_analysis = missing_analysis[missing_analysis['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("Columns with missing values:")
print(missing_analysis.to_string(index=False))

# buang sng missing lebih dari 70 karena ya bad aja kurang berguna datae takut bias
high_missing_cols = missing_analysis[missing_analysis['missing_pct'] > 70]['column'].tolist()
if high_missing_cols:
    combined_df = combined_df.drop(columns=high_missing_cols)
    print(f"\nDropped {len(high_missing_cols)} columns with >70% missing data:")
    print(high_missing_cols)

# nyoba ngisi missing value
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = combined_df.select_dtypes(include=['object', 'bool']).columns.tolist()

# Numerical imputation
for col in numerical_cols:
    if combined_df[col].isnull().any():
        skewness = combined_df[col].skew()
        if abs(skewness) > 2:  # Highly skewed pake median -> tujuan biar puncak e ga geser jauh
            fill_val = combined_df[col].median()
        else: # kalau udah terdistribusi baik ya tinggal diisi mean aja
            fill_val = combined_df[col].mean()
        combined_df[col] = combined_df[col].fillna(fill_val)

# Categorical imputation
for col in categorical_cols:
    if combined_df[col].isnull().any():
        mode_vals = combined_df[col].mode()
        fill_val = mode_vals[0] if len(mode_vals) > 0 else 'unknown'
        combined_df[col] = combined_df[col].fillna(fill_val)

print("Missing value imputation complete.")

# -----------------------------------------------------------------
# 4. Feature Selection and Preprocessing
# -----------------------------------------------------------------
print("\nStep 4: Feature Selection...")
# buang fitur-fitur yang variasinya dikit
from sklearn.feature_selection import VarianceThreshold
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    selector = VarianceThreshold(threshold=0.01)
    low_var_mask = selector.fit_transform(combined_df[numerical_cols])
    low_var_features = numerical_cols[~selector.get_support()]
    if len(low_var_features) > 0:
        combined_df = combined_df.drop(columns=low_var_features)
        print(f"Removed {len(low_var_features)} low-variance features")

# buang yang korelasi tinggi diatas 85%
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    corr_matrix = combined_df[numerical_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.85)]
    
    if high_corr_features:
        combined_df = combined_df.drop(columns=high_corr_features)
        print(f"Removed {len(high_corr_features)} highly correlated features")

# One-hot encode semua data categorical
categorical_cols = combined_df.select_dtypes(include=['object', 'bool']).columns.tolist()
if categorical_cols:
    print(f"One-hot encoding {len(categorical_cols)} categorical features")
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# ngerapiin nama header/column
combined_df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in combined_df.columns]
combined_df.columns = [re.sub(r'_+', '_', col).strip('_') for col in combined_df.columns]

# handle outlier pake IQR method alesan e biar ga terlalu skewed aja in case ada sng skewed parah
numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
outlier_count = 0
for col in numerical_cols:
    Q1, Q3 = combined_df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_before = ((combined_df[col] < lower_bound) | (combined_df[col] > upper_bound)).sum()
    if outliers_before > 0:
        combined_df[col] = np.clip(combined_df[col], lower_bound, upper_bound)
        outlier_count += outliers_before

print(f"Handled {outlier_count} outliers across all features")
print(f"Final feature count: {combined_df.shape[1]}")

# -----------------------------------------------------------------
# 5. Comprehensive Model Selection and Training
# -----------------------------------------------------------------
print("\nStep 5: Model Selection and Training...")
train_len = len(train_y)
X_train = combined_df.iloc[:train_len].copy()
X_test = combined_df.iloc[train_len:].copy()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# cv ni cross val
def get_models_for_cv():
    """Returns models for cross_val_score (no early stopping)"""
    return {
        'lightgbm': lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, verbose=-1),
        'xgboost': xgb.XGBRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, verbosity=0),
        'catboost': cb.CatBoostRegressor(random_state=42, iterations=1000, learning_rate=0.05, verbose=0),
        'random_forest': RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
        'extra_trees': ExtraTreesRegressor(random_state=42, n_estimators=200, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1),
        'ridge': RidgeCV(alphas=np.logspace(-4, 4, 20)),
        'elastic_net': ElasticNetCV(alphas=np.logspace(-4, 4, 10), l1_ratio=[0.1, 0.5, 0.9], cv=5),
        'lasso': LassoCV(alphas=np.logspace(-4, 4, 20), cv=5)
    }

# ini untuk yang stacking
def get_models_for_stacking():
    """Returns models for stacking loop (with early stopping where applicable)"""
    return {
        'lightgbm': lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, verbose=-1),
        'xgboost': xgb.XGBRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, verbosity=0, early_stopping_rounds=50),
        'catboost': cb.CatBoostRegressor(random_state=42, iterations=1000, learning_rate=0.05, verbose=0, early_stopping_rounds=50),
        'random_forest': RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
        'extra_trees': ExtraTreesRegressor(random_state=42, n_estimators=200, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    }

# Cross-validation setup
NFOLDS = 5
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Single model evaluation -> CV
print("\n" + "="*60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*60)

models_for_cv = get_models_for_cv()
model_scores = {}

for name, model in models_for_cv.items():
    print(f"\nEvaluating {name}...")
    cv_scores = cross_val_score(model, X_train, train_y, cv=kfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    model_scores[name] = {'cv_rmse': cv_rmse, 'cv_std': cv_std}
    print(f"  CV RMSE: {cv_rmse:.4f} (±{cv_std:.4f})")
print(f"\n{'Model':<20} {'CV RMSE':<10} {'Std':<10}")
print("-" * 40)
for name, scores in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse']):
    print(f"{name:<20} {scores['cv_rmse']:<10.4f} {scores['cv_std']:<10.4f}")

# -----------------------------------------------------------------
# 6. Model Building: Out-of-Fold (OOF) Stacking
# -----------------------------------------------------------------
print("\nStep 6: Building and Training Models using Out-of-Fold Stacking...")
train_len = len(train_y)
X_train = combined_df.iloc[:train_len].copy()
X_test = combined_df.iloc[train_len:].copy()

# Cross-validation setup
NFOLDS = 5
tsfold = TimeSeriesSplit(n_splits=NFOLDS)

def objective_lgb(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1),
        'num_leaves': trial.suggest_int("num_leaves", 2, 256),
        'max_depth': trial.suggest_int("max_depth", -1, 50),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'bagging_freq': trial.suggest_int("bagging_freq", 1, 10),
        'bagging_fraction': trial.suggest_float("bagging_fraction", 0.5, 1.0),
        'feature_fraction': trial.suggest_float("feature_fraction", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = Pipeline([
        ('regressor', lgb.LGBMRegressor(**params))
    ])
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()


def objective_xgb(trial):
    params = {
        'n_estimators': 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0
    }
    model = Pipeline([
        ('regressor', xgb.XGBRegressor(**params))
    ])
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()


def objective_cat(trial):
    params = {
        'iterations': 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True),
        'verbose': 0,
        'random_state': 42
    }
    model = Pipeline([
        ('regressor', cb.CatBoostRegressor(**params))
    ])
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()


def objective_random_forest(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 500),
        'max_depth': trial.suggest_int("max_depth", 5, 30),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'random_state': 42,
        'n_jobs': -1
    }
    model = Pipeline([
        ('regressor', RandomForestRegressor(**params))
    ])
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()


def objective_extra_trees(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 500),
        'max_depth': trial.suggest_int("max_depth", 5, 30),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 20),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'random_state': 42,
        'n_jobs': -1
    }
    model = Pipeline([
        ('regressor', ExtraTreesRegressor(**params))
    ])
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()


def objective_gradient_boosting(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 500),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'random_state': 42
    }
    model = Pipeline([
        ('regressor', GradientBoostingRegressor(**params))
    ])
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold,
                                scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -cv_scores.mean()

    

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(lambda trial: objective_lgb(trial),
                   n_trials=25, show_progress_bar=True)

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(lambda trial: objective_xgb(trial),
                   n_trials=25, show_progress_bar=True)

study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(lambda trial: objective_cat(trial),
                   n_trials=25, show_progress_bar=True)

study_rf = optuna.create_study(direction="minimize")
study_rf.optimize(lambda trial: objective_random_forest(trial),
                  n_trials=25, show_progress_bar=True)

study_et = optuna.create_study(direction="minimize")
study_et.optimize(lambda trial: objective_extra_trees(trial),
                  n_trials=25, show_progress_bar=True)

study_gb = optuna.create_study(direction="minimize")
study_gb.optimize(lambda trial: objective_gradient_boosting(trial),
                  n_trials=25, show_progress_bar=True)

best_params = {
    "lightgbm": study_lgb.best_trial.params,
    "xgboost": study_xgb.best_trial.params,
    "catboost": study_cat.best_trial.params,
    "random_forest": study_rf.best_trial.params,
    "extra_trees": study_et.best_trial.params,
    'gradient_boosting': study_gb.best_trial.params
}

def get_models():
    """Return dictionary of models with optimized hyperparameters"""
    return {
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
        
        # Linear models 
        'ridge': RidgeCV(alphas=np.logspace(-4, 4, 20)),
        'elastic_net': ElasticNetCV(alphas=np.logspace(-4, 4, 10), 
                                   l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5),
        'lasso': LassoCV(alphas=np.logspace(-4, 4, 20), cv=5)
    }

# Single model evaluation
print("\n" + "="*60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*60)

models = get_models()
model_scores = {}
model_predictions = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    cv_scores = cross_val_score(model, X_train, train_y, cv=tsfold, 
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    model_scores[name] = {'cv_rmse': cv_rmse, 'cv_std': cv_std}
    
    model.fit(X_train, train_y)
    test_pred = model.predict(X_test)
    model_predictions[name] = test_pred
    
    print(f"  CV RMSE: {cv_rmse:.4f} (±{cv_std:.4f})")

print(f"\n{'Model':<15} {'CV RMSE':<10} {'Std':<10}")
print("-" * 35)
for name, scores in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse']):
    print(f"{name:<15} {scores['cv_rmse']:<10.4f} {scores['cv_std']:<10.4f}")

# -----------------------------------------------------------------
# 6. Advanced Ensemble Methods
# -----------------------------------------------------------------
print("\n" + "="*60)
print("ENSEMBLE METHODS")
print("="*60)

# OOF prediction untuk stacking
print("\nGenerating out-of-fold predictions...")
selected_model_names = [name for name, _ in sorted(model_scores.items(), key=lambda x: x[1]['cv_rmse'])[:5]]
print(f"Selected top 5 models: {selected_model_names}")

oof_predictions = np.zeros((len(X_train), len(selected_model_names)))
test_predictions = np.zeros((len(X_test), len(selected_model_names)))

models_for_stacking = get_models()

for i, name in enumerate(selected_model_names):
    print(f"--- Stacking training for {name} ---")
    model = models_for_stacking[name]
    fold_test_preds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        
        # untuk early stopping gradient boosting
        if name in ['lightgbm', 'xgboost', 'catboost']:
            model.fit(X_fold_train, y_fold_train,
                     eval_set=[(X_fold_val, y_fold_val)])
        else:
            model.fit(X_fold_train, y_fold_train)
        
        oof_predictions[val_idx, i] = model.predict(X_fold_val)
        fold_test_preds.append(model.predict(X_test))
    
    test_predictions[:, i] = np.mean(fold_test_preds, axis=0)

# -----------------------------------------------------------------
# 7. Meta-Model Training and Final Prediction
# -----------------------------------------------------------------
print("\nTraining meta-models...")
meta_models = {
    'ridge_meta': RidgeCV(alphas=np.logspace(-4, 4, 20)),
    'elastic_meta': ElasticNetCV(alphas=np.logspace(-4, 4, 10), 
                                l1_ratio=[0.1, 0.5, 0.7, 0.9])
}

best_meta_rmse = float('inf')
best_meta_model = None
best_meta_name = None

for meta_name, meta_model in meta_models.items():
    meta_model.fit(oof_predictions, train_y)
    meta_pred = meta_model.predict(oof_predictions)
    meta_rmse = np.sqrt(mean_squared_error(train_y, meta_pred))
    
    print(f"{meta_name}: {meta_rmse:.4f}")
    
    if meta_rmse < best_meta_rmse:
        best_meta_rmse = meta_rmse
        best_meta_model = meta_model
        best_meta_name = meta_name

print(f"\nBest meta-model: {best_meta_name} (RMSE: {best_meta_rmse:.4f})")

final_predictions = best_meta_model.predict(test_predictions)
final_predictions = np.maximum(final_predictions, 0)

# -----------------------------------------------------------------
# 8. Create Submission
# -----------------------------------------------------------------
print("\nStep 8: Creating Submission...")

submission_df = pd.DataFrame({
    'date': submission_dates,
    'daily_ktCO2': final_predictions
})

submission_df.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")
print(f"\nFirst 5 predictions:")
print(submission_df.head())

print(f"\nTraining completed! Final OOF RMSE: {best_meta_rmse:.4f}")
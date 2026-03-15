# Daily CO₂ Emission Prediction

**Project for RASIO 2025 Competition Data Science Category by Padjadjaran University**
**[View the Project Presentation Here]([https://drive.google.com/file/d/1u-3Boneuk0-eehA5zWctDK4YOaEsjhIO/view?usp=sharing])**

## Overview
As the industrial sector continues to grow rapidly, it becomes increasingly important to accurately predict daily CO₂ emission levels. Reliable forecasts are essential for developing effective environmental policies and public health strategies. This project was developed as part of a data science competition, leveraging advanced data science methodologies to build a robust predictive model for daily CO₂ emissions.

## Methodology

### Data Preprocessing & Feature Engineering
Comprehensive data preprocessing was performed to clean and prepare the dataset for modeling. A key component of our approach was feature engineering, where we extracted and created various time-based features. These features were critical in allowing the models to capture underlying temporal patterns and seasonal trends in the emission data.

### Modeling Strategy: Out-of-Fold Stacking
To maximize predictive accuracy and ensure strong generalization, we implemented an out-of-fold stacking ensemble approach. 

#### Base Models
We initially experimented with nine different algorithms: CatBoost, XGBoost, Gradient Boosting, LightGBM, Random Forest, Extra Trees, Ridge, Lasso, and Elastic Net. Based on their individual performances, we excluded Lasso and Elastic Net, moving forward with the top 7 models as our base learners:
1. CatBoost
2. XGBoost
3. Gradient Boosting
4. LightGBM
5. Random Forest
6. Extra Trees
7. Ridge Regression

#### Meta-Learner
The predictions from the 7 base models were combined using a meta-learner. We tested several algorithms for this role, including ElasticNet, Ridge, Lasso, and Multi-Layer Perceptron (MLP). **Ridge Regression** achieved the best performance as the meta-model and was selected for the final architecture.

## Results & Performance

Our stacked ensemble model demonstrated strong generalization performance and highly reliable predictive capability, earning a top-tier placement in the competition.

* **Training RMSE:** 1.6933
* **Private Leaderboard RMSE:** 1.953
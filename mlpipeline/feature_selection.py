# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# """
# This script identifies selected features from baseline models
# while printing evaluation metrics for traceability.
# """

# def build_baseline_models(df):
#     """
#     Builds baseline models to select important features.

#     Args:
#         df (pd.DataFrame): Prepared dataset with features and target.

#     Returns:
#         dict: Selected features per model.
#     """
#     features = [
#         'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)',
#         'Speed', 'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
#         'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
#         'Weekday_sin', 'Weekday_cos', 'MinutesSinceSunrise', 'MinutesUntilSunset'
#     ]

#     X = df[features]
#     y = df['Radiation']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     models = {
#         'RandomForest': RandomForestRegressor(),
#         'XGBoost': XGBRegressor(),
#         'CatBoost': CatBoostRegressor(verbose=0)
#     }

#     feature_threshold = 0.005
#     selected_features_per_model = {}

#     for name, model in models.items():
#         model.fit(X_train, y_train)

#         # Feature importance extraction
#         if name == 'XGBoost':
#             importances = pd.Series(model.get_booster().get_score(importance_type='weight'))
#             importances = importances / importances.sum()
#             importances = importances.reindex(X.columns, fill_value=0)
#         else:
#             importances = pd.Series(model.feature_importances_, index=X.columns)

#         selected_features = importances[importances >= feature_threshold].index.tolist()
#         selected_features_per_model[name] = selected_features

#         # Evaluate and print metrics
#         preds = model.predict(X_test)
#         rmse = np.sqrt(mean_squared_error(y_test, preds))
#         r2 = r2_score(y_test, preds)
#         print(f"\nModel: {name}")
#         print(f"Selected Features ({len(selected_features)}): {selected_features}")
#         print(f"RMSE: {rmse:.4f}")
#         print(f"R2: {r2:.4f}")

#     return selected_features_per_model

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from prefect import task, get_run_logger


def build_baseline_models(df):
    logger = get_run_logger()
    logger.info("Starting baseline model building")

    features = [
        'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)',
        'Speed', 'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
        'Day_sin', 'Day_cos', 'Month_sin', 'Month_cos',
        'Weekday_sin', 'Weekday_cos', 'MinutesSinceSunrise', 'MinutesUntilSunset'
    ]

    X = df[features]
    y = df['Radiation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(),
        'XGBoost': XGBRegressor(),
        'CatBoost': CatBoostRegressor(verbose=0)
    }

    feature_threshold = 0.005
    selected_features_per_model = {}

    for name, model in models.items():
        logger.info(f"Training {name} model")
        model.fit(X_train, y_train)

        if name == 'XGBoost':
            importances = pd.Series(model.get_booster().get_score(importance_type='weight'))
            importances = importances / importances.sum()
            importances = importances.reindex(X.columns, fill_value=0)
        else:
            importances = pd.Series(model.feature_importances_, index=X.columns)

        selected_features = importances[importances >= feature_threshold].index.tolist()
        selected_features_per_model[name] = selected_features

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        logger.info(f"{name} - Selected Features ({len(selected_features)}): {selected_features}")
        logger.info(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

    logger.info("Baseline model building completed")
    return selected_features_per_model

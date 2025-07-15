# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# import numpy as np

# """
# Train and tune models based on selected features.
# """

# def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
#     """
#     Split dataset into train, validation, and test sets.

#     Returns:
#         X_train, X_val, X_test, y_train, y_val, y_test
#     """
#     X = df[[col for col in df.columns if col != 'Radiation']]
#     y = df['Radiation']

#     X_train_full, X_test, y_train_full, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )

#     val_relative_size = val_size / (1 - test_size)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_full, y_train_full, test_size=val_relative_size, random_state=random_state
#     )

#     return X_train, X_val, X_test, y_train, y_val, y_test


# def get_model_instance(model_name, params):
#     """
#     Returns a model instance given the name and parameters.
#     """
#     if model_name == 'RandomForest':
#         return RandomForestRegressor(**params)
#     elif model_name == 'XGBoost':
#         return XGBRegressor(**params)
#     elif model_name == 'CatBoost':
#         return CatBoostRegressor(**params)
#     else:
#         raise ValueError(f"Unsupported model name: {model_name}")


# def evaluate_model(model, X_val, y_val):
#     """
#     Evaluate the model on validation data.

#     Returns:
#         tuple: (RMSE, R2)
#     """
#     preds = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, preds))
#     r2 = r2_score(y_val, preds)
#     return rmse, r2


# def train_tune_models(df, selected_features_per_model):
#     """
#     Tune multiple models using different hyperparameters.

#     Args:
#         df (pd.DataFrame): Dataset.
#         selected_features_per_model (dict): Features selected per model.

#     Returns:
#         all_runs (list): Run summaries.
#         X_test (pd.DataFrame)
#         y_test (pd.Series)
#     """
#     param_grids = {
#         'RandomForest': [
#             {}, 
#             {'n_estimators': 150, 'max_depth': None},
#             {'n_estimators': 300, 'max_depth': 15},
#         ],
#         'XGBoost': [
#             {},
#             {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
#             {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.03},
#         ],
#         'CatBoost': [
#             {'verbose': 0}, 
#             {'iterations': 200, 'depth': 5, 'learning_rate': 0.05, 'verbose': 0},
#             {'iterations': 300, 'depth': 6, 'learning_rate': 0.03, 'verbose': 0},
#         ]
#     }

#     X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
#     all_runs = []

#     for model_name, param_list in param_grids.items():
#         features = selected_features_per_model[model_name]
#         for params in param_list:
#             model = get_model_instance(model_name, params)
#             model.fit(X_train[features], y_train)

#             val_rmse, val_r2 = evaluate_model(model, X_val[features], y_val)

#             all_runs.append({
#                 'model_name': model_name,
#                 'params': params,
#                 'features': features,
#                 'val_rmse': val_rmse,
#                 'val_r2': val_r2,
#                 'model': model
#             })

#     return all_runs, X_val, X_test, y_test

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np
from prefect import task, get_run_logger



def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    logger = get_run_logger()
    logger.info("Splitting dataset into train, validation, and test sets")
    
    X = df[[col for col in df.columns if col != 'Radiation']]
    y = df['Radiation']

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_relative_size, random_state=random_state
    )

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model_instance(model_name, params):
    if model_name == 'RandomForest':
        return RandomForestRegressor(**params)
    elif model_name == 'XGBoost':
        return XGBRegressor(**params)
    elif model_name == 'CatBoost':
        return CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    return rmse, r2


@task(task_run_name="train and tune_models")
def train_tune_models(df, selected_features_per_model):
    logger = get_run_logger()
    logger.info("Starting training and tuning models")

    param_grids = {
        'RandomForest': [
            {}, 
            {'n_estimators': 150, 'max_depth': None},
            {'n_estimators': 300, 'max_depth': 15},
        ],
        'XGBoost': [
            {},
            {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.03},
        ],
        'CatBoost': [
            {'verbose': 0}, 
            {'iterations': 200, 'depth': 5, 'learning_rate': 0.05, 'verbose': 0},
            {'iterations': 300, 'depth': 6, 'learning_rate': 0.03, 'verbose': 0},
        ]
    }

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    all_runs = []

    for model_name, param_list in param_grids.items():
        features = selected_features_per_model[model_name]
        logger.info(f"Training {model_name} with {len(features)} features")

        for params in param_list:
            model = get_model_instance(model_name, params)
            model.fit(X_train[features], y_train)

            val_rmse, val_r2 = evaluate_model(model, X_val[features], y_val)

            logger.info(f"{model_name} | Params: {params} | RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")

            all_runs.append({
                'model_name': model_name,
                'params': params,
                'features': features,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'model': model
            })

    logger.info("Training and tuning completed")
    return all_runs, X_val, X_test, y_test

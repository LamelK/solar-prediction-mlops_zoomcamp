from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from prefect import task, get_run_logger


def get_model_instance(model_name, params):
    """
    Return an instance of the specified regression model with given parameters.
    Applies scaling for KNN models.
    """
    if model_name == "RandomForest":
        return RandomForestRegressor(**params)
    elif model_name == "GradientBoosting":
        return GradientBoostingRegressor(**params)
    elif model_name == "KNN":
        # Always apply scaling for KNN
        knn = KNeighborsRegressor(**params)
        return make_pipeline(StandardScaler(), knn)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_model(model, X_val, y_val):
    """
    Compute RMSE and R2 metrics for a model on validation data.
    """
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    return rmse, r2


@task(name="Train and Tune Models")
def train_tune_models(df):
    """
    Train and tune multiple regression models using predefined hyperparameters.
    Returns all runs and validation/test splits.
    """
    logger = get_run_logger()
    logger.info("Starting model training and hyperparameter tuning")

    # Define hyperparameter grids for each model
    param_grids = {
        "RandomForest": [{"n_estimators": 300, "max_depth": 15}, ],
        "GradientBoosting": [
            {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        ],
        "KNN": [{"n_neighbors": 5, "weights": "uniform"}, ],
    }

    # Split features and target
    X = df.drop("Radiation", axis=1)
    y = df["Radiation"]
    # Split data into train, validation, and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15 / 0.85, random_state=42
    )

    logger.info(
        f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    all_runs = []

    for model_name, param_list in param_grids.items():
        for params in param_list:
            logger.info(f"Training {model_name} with params: {params}")
            model = get_model_instance(model_name, params)
            model.fit(X_train, y_train)

            # Evaluate on validation set
            val_rmse, val_r2 = evaluate_model(model, X_val, y_val)
            logger.info(
                f"{model_name} | Params: {params} | Val RMSE: {val_rmse:.4f}, "
                f"Val R2: {val_r2:.4f}"
            )

            all_runs.append(
                {
                    "model_name": model_name,
                    "params": params,
                    "features": X.columns.tolist(),
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                    "model": model,
                }
            )

    logger.info("Model training and tuning completed")
    return all_runs, X_val, X_test, y_test

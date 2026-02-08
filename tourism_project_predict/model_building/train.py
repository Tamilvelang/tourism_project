
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib
from huggingface_hub import HfApi

# --- 1. PREPARE DATA ---

X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# Label Encoding for categorical features
cat_cols = X_train.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# --- 2. UPDATED TRAINING FUNCTION ---
mlflow.set_experiment("Wellness_Tourism_XGBoost")

def train_and_log(max_depth, learning_rate, n_estimators):
    with mlflow.start_run(run_name=f"XGB_d{max_depth}_lr{learning_rate}"):
        # Removed 'use_label_encoder' to fix Warning
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "random_state": 42,
            "eval_metric": "logloss"
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        # Updated 'name' instead of 'artifact_path' for MLflow 3.0+
        mlflow.xgboost.log_model(model, name="model")

        print(f"Depth {max_depth} | Accuracy: {acc:.4f} | Recall: {rec:.4f}")
        return model, rec

# --- 3. TUNING LOOP ---
param_grids = [(3, 0.1, 100), (5, 0.05, 200), (7, 0.01, 300)]
best_model, best_recall = None, 0

for depth, lr, n_est in param_grids:
    model, recall = train_and_log(depth, lr, n_est)
    if recall > best_recall:
        best_recall = recall
        best_model = model

# --- 4. REGISTER MODEL ---
model_save_path = "best_xgboost_model.joblib"
joblib.dump(best_model, model_save_path)

MODEL_REPO_ID = "Tamilvelan/tourism-wellness-model"

api = HfApi()

api.upload_file(
    path_or_fileobj=model_save_path,
    path_in_repo="model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)
print(f"Model successfully registered at {MODEL_REPO_ID}")

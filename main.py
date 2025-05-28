from sklearn.datasets import load_iris
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample dataset (Iris)
data = load_iris(as_frame=True)
df = data.frame
target_column = "target"

# Split data
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Phase 2: Ask for Constraints ===
time_budget = int(input("⏱️ How much time (in seconds) can we train the model? "))
want_interpretable = input("🧠 Do you prefer simple, interpretable models? (yes/no): ").strip().lower() == "yes"

# === Phase 2: Choose Models Based on Constraints ===

if want_interpretable:
    estimator_list = ["lrl1", "rf", "extra_tree"]
else:
    estimator_list = ["lgbm", "xgboost", "rf", "extra_tree", "catboost"]

# Run FLAML AutoML
automl = AutoML()
automl_settings = {
    "time_budget": time_budget,
    "metric": "accuracy",
    "task": "classification",
    "log_file_name": "acaml.log",
    "estimator_list": estimator_list
}
automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

# Evaluate
y_pred = automl.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Best Model:", automl.model)

# === SHAP Explainability ===
import shap
import matplotlib.pyplot as plt

try:
    # Extract the core model (FLAML wraps it)
    core_model = automl.model.estimator if hasattr(automl.model, 'estimator') else automl.model

    # Create SHAP explainer for compatible models
    explainer = shap.Explainer(core_model, X_train)
    shap_values = explainer(X_test)

    print("\n🧠 Feature Importance (SHAP): Launching summary plot...")
    shap.summary_plot(shap_values, X_test, show=True)

except Exception as e:
    print("⚠️ Could not generate SHAP explanation:", e)


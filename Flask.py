import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier    
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# === Load Dataset ===
df = pd.read_csv('heart.csv')  # Replace with your actual CSV file name

# === Features and Target ===
X = df.drop('target', axis=1)
y = df['target']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === XGBoost Model ===
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

# === MLP Model ===
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]

# === Evaluation Function ===
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"----- {name} -----")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("ROC AUC  :", roc_auc_score(y_true, y_prob))
    print()

# === Evaluate Both Models ===
evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)
evaluate_model("MLP", y_test, y_pred_mlp, y_prob_mlp)

# === Plot ROC Curves ===
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
plt.plot(fpr_mlp, tpr_mlp, label='MLP')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()



# === Runtime Input and Prediction ===
print("\n=== Runtime Prediction ===")
feature_names = X.columns.tolist()
input_values = []

print("Please enter the following values:")

for feature in feature_names:
    val = float(input(f"{feature}: "))
    input_values.append(val)

# Convert to numpy array and scale
input_array = np.array(input_values).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Predict using both models
runtime_pred_xgb = xgb.predict(input_scaled)[0]
runtime_pred_mlp = mlp.predict(input_scaled)[0]

# Output prediction
print("\n--- Prediction Result ---")
print(f"XGBoost Prediction: {'Heart Disease' if runtime_pred_xgb == 1 else 'No Heart Disease'}")
print(f"MLP Prediction    : {'Heart Disease' if runtime_pred_mlp == 1 else 'No Heart Disease'}")

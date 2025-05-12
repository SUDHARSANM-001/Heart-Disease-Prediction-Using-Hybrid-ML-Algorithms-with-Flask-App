import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = 'secret'  # For session management

# === Load & Preprocess Data ===
df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']
features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Models ===
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# === Evaluation ===
y_pred_xgb = xgb.predict(X_test_scaled)
y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

y_pred_mlp = mlp.predict(X_test_scaled)
y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
roc_xgb = roc_auc_score(y_test, y_prob_xgb)
roc_mlp = roc_auc_score(y_test, y_prob_mlp)

# === Plot ROC Curve ===
def plot_roc_curve():
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)

    plt.figure()
    plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
    plt.plot(fpr_mlp, tpr_mlp, label='MLP')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/graph.png')
    plt.close()

plot_roc_curve()

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['user'] = 'admin'
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/graph')
def graph():
    df = pd.read_csv('heart.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]

    # === Evaluation Metrics ===
    metrics = {
        "acc_xgb": accuracy_score(y_test, y_pred_xgb),
        "prec_xgb": precision_score(y_test, y_pred_xgb),
        "recall_xgb": recall_score(y_test, y_pred_xgb),
        "f1_xgb": f1_score(y_test, y_pred_xgb),
        "acc_mlp": accuracy_score(y_test, y_pred_mlp),
        "prec_mlp": precision_score(y_test, y_pred_mlp),
        "recall_mlp": recall_score(y_test, y_pred_mlp),
        "f1_mlp": f1_score(y_test, y_pred_mlp)
    }

    # === ROC Curve Plot ===
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)

    plt.figure()
    plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')
    plt.plot(fpr_mlp, tpr_mlp, label='MLP')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()

    graph_path = os.path.join('static', 'roc_curve.png')
    plt.savefig(graph_path)
    plt.close()

    return render_template('graph.html', graph_url=graph_path, **metrics)


 @app.route('/prediction', methods=['GET', 'POST'])
 def prediction():
     if 'user' not in session:
         return redirect(url_for('login'))
     if request.method == 'POST':
         try:
             input_values = [float(request.form[feature]) for feature in features]
             input_array = np.array(input_values).reshape(1, -1)
             input_scaled = scaler.transform(input_array)
             pred_xgb = xgb.predict(input_scaled)[0]
             pred_mlp = mlp.predict(input_scaled)[0]
             result_xgb = 'Heart Disease' if pred_xgb == 1 else 'No Heart Disease'
             result_mlp = 'Heart Disease' if pred_mlp == 1 else 'No Heart Disease'
             return render_template('prediction.html', features=features, result_xgb=result_xgb, result_mlp=result_mlp)
         except Exception as e:
             return str(e)
     return render_template('prediction.html', features=features)


 @app.route('/prediction', methods=['GET', 'POST'])
 def prediction():
     if 'user' not in session:
         return redirect(url_for('login'))

     if request.method == 'POST':
         try:
             patient_name = request.form['patient_name']
             input_dict = {feature: float(request.form[feature]) for feature in features}
             input_array = np.array(list(input_dict.values())).reshape(1, -1)
             input_scaled = scaler.transform(input_array)

             pred_xgb = xgb.predict(input_scaled)[0]
             pred_mlp = mlp.predict(input_scaled)[0]

             result_xgb = 'Heart Disease' if pred_xgb == 1 else 'No Heart Disease'
             result_mlp = 'Heart Disease' if pred_mlp == 1 else 'No Heart Disease'

             return render_template('prediction.html',
                                    features=features,
                                    input_dict=input_dict,
                                    result_xgb=result_xgb,
                                    result_mlp=result_mlp,
                                    patient_name=patient_name)
         except Exception as e:
             return str(e)

     return render_template('prediction.html', features=features)







@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            patient_name = request.form['patient_name']
            input_dict = {feature: float(request.form[feature]) for feature in features}
            input_array = np.array(list(input_dict.values())).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Get prediction probabilities if available
            prob_xgb = xgb.predict_proba(input_scaled)[0][1]  # Probability of class 1
            prob_mlp = mlp.predict_proba(input_scaled)[0][1]  # Probability of class 1

            # Final percentage as average of both
            final_prob = (prob_xgb + prob_mlp) / 2
            final_percentage = round(final_prob * 100, 2)

            result_xgb = 'Heart Disease' if prob_xgb >= 0.5 else 'No Heart Disease'
            result_mlp = 'Heart Disease' if prob_mlp >= 0.5 else 'No Heart Disease'

            return render_template('prediction.html',
                                   features=features,
                                   input_dict=input_dict,
                                   result_xgb=result_xgb,
                                   result_mlp=result_mlp,
                                   final_percentage=final_percentage,
                                   patient_name=patient_name)
        except Exception as e:
            return str(e)

    return render_template('prediction.html', features=features)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=False, port=8000)

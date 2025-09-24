# compare_accuracy.py
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Data_Utils.utils import recup_data_csv
from Data_Utils.math import sigmoid

TRAIN_FILE = "../datasets/dataset_train.csv"


# ---------------------------------------------------------------------------
# [1] Fonctions pour ton modèle maison
def gradient_descent(X, y, alpha=0.1, iterations=5000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta -= alpha * gradient
    return theta


def train_one_vs_all(X, y, labels, alpha=0.1, iterations=5000):
    all_theta = {}
    for label in labels:
        y_binary = (y == label).astype(int)
        theta = gradient_descent(X, y_binary, alpha, iterations)
        all_theta[label] = theta
    return all_theta


def predict_one_vs_all(X, all_theta):
    preds = []
    for x in X:
        scores = {label: sigmoid(x.dot(theta)) for label, theta in all_theta.items()}
        preds.append(max(scores, key=scores.get))
    return preds

# ---------------------------------------------------------------------------
def main():
    # Charger les données
    df = recup_data_csv(TRAIN_FILE)
    features = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Charms", "Flying"]
    X = df[features].fillna(0)
    y = df["Hogwarts House"]

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]  # ajout biais

    labels = y.unique()

    # ================== Ton modèle maison ==================
    thetas = train_one_vs_all(X_bias, y.values, labels, alpha=0.3, iterations=10_000)
    y_pred_maison = predict_one_vs_all(X_bias, thetas)
    acc_maison = accuracy_score(y, y_pred_maison)

    # ================== Scikit-Learn ==================
    clf = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10_000)
    clf.fit(X_scaled, y)
    y_pred_sklearn = clf.predict(X_scaled)
    acc_sklearn = accuracy_score(y, y_pred_sklearn)

    # ================== Résultats ==================
    print("Comparaison des modèles :")
    print(f" - Accuracy modèle maison     : {acc_maison:.4f}")
    print(f" - Accuracy Scikit-Learn      : {acc_sklearn:.4f}")


if __name__ == "__main__":
    main()

# logreg_sklearn.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from Data_Utils.utils import recup_data_csv

# =============================== PARAMÈTRES ===================================
TRAIN_FILE = "datasets/dataset_train.csv"
TEST_FILE = "datasets/dataset_test.csv"

# =============================== MAIN =========================================
def main():
    # [1] Charger les données
    df_train = recup_data_csv(TRAIN_FILE)
    df_test = recup_data_csv(TEST_FILE)

    # [2] Features identiques à ton implémentation
    features = ["Herbology", "Defense Against the Dark Arts"]
    X_train = df_train[features].fillna(0)
    y_train = df_train["Hogwarts House"]

    X_test = df_test[features].fillna(0)

    # [3] Normalisation (comme tu le fais dans ton code)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # [4] Modèle LogisticRegression scikit-learn
    clf = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10000)
    clf.fit(X_train_scaled, y_train)

    # [5] Prédiction
    y_pred_train = clf.predict(X_train_scaled)

    # Évaluer la précision sur les données d'entraînement
    acc = accuracy_score(y_train, y_pred_train)
    print(f"Accuracy sur dataset_train : {acc:.4f}")

    # Sauvegarde des prédictions sur dataset_test
    y_pred_test = clf.predict(X_test_scaled)
    result = pd.DataFrame({
        "Index": range(len(y_pred_test)),
        "Hogwarts House": y_pred_test
    })
    result.to_csv("houses_sklearn.csv", index=False)
    print("Prédictions sauvegardées dans houses_sklearn.csv")

if __name__ == "__main__":
    main()

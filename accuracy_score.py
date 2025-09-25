import numpy as np
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Data_Utils.utils import recup_data_csv
from Data_Utils.math import sigmoid
from Data_Utils.logger import setup_logger

# =============================== CONSTANTES ===================================
LOG = setup_logger()


# =============================== FONCTIONS ====================================
def accuracy_score(y_true, y_pred):
    """Calcule le pourcentage de prédictions correctes."""
    return (y_true == y_pred).mean()

# ------------------------------------------------------------------------------
def predict_house(X, file):
    with open(file, "r") as f:
        thetas = json.load(f)

    # ajout de la colonne de biais
    X.insert(0, "bias", 1)
    X_values = X.values
    prediction_tab = []

    houses = list(thetas.keys())
    for x in X_values:
        scores = {}
        for house in houses:
            theta = np.array(thetas[house])
            scores[house] = sigmoid(np.dot(x, theta))
        predicted_house = max(scores, key=scores.get)
        prediction_tab.append(predicted_house)

    return prediction_tab


# ------------------------------------------------------------------------------
def main() -> int:
    try:
        # [1]. Récupération des données TRAIN (pour accuracy)
        df_train = recup_data_csv("datasets/dataset_train.csv")

        # Utiliser exactement les mêmes features qu'à l'entraînement :
        features = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Charms", "Flying"]
        X_train = df_train[features].fillna(0)

        # Appliquer la même normalisation que dans logreg_train.py :
        X_train = (X_train - X_train.mean()) / X_train.std()

        y_true = df_train["Hogwarts House"]

        # [2]. Prédictions sur TRAIN
        y_pred = predict_house(X_train.copy(), "weights.json")

        # [3]. Accuracy
        acc = accuracy_score(y_true.values, np.array(y_pred))
        LOG.info(f"Precision sur dataset_train.csv : {acc:.4f} soit {acc * 100}%")

    except FileNotFoundError as e:
        LOG.critical(f"Erreur CRITIQUE - {e}")
        return 2

    except Exception as e:
        LOG.critical(f"Erreur CRITIQUE - {e}")
        return 1

    finally:
        LOG.info("Fermeture du programme !")
        LOG.info("-----------------------------------------------------------------")
    return 0


# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())

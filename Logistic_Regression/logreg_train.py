import pandas as pd
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
def gradient_descent(X, y, alpha=0.1, iterations=5000):
    """
    Effectue la descente de gradient pour la régression logistique.

    Args:
        X (numpy.ndarray): Matrice des features de forme (m, n), avec m exemples et n features (incluant la colonne de biais).
        y (numpy.ndarray): Vecteur des labels binaires de forme (m, ).
        alpha (float, optional): Taux d'apprentissage. Défaut à 0.1.
        iterations (int, optional): Nombre d'itérations pour la descente de gradient. Défaut à 1000.

    Returns:
        numpy.ndarray: Vecteur des poids theta de forme (n, ), optimisés pour minimiser la fonction de coût.
    
    Description:
        La fonction initialise un vecteur theta à zéro, puis met à jour theta à chaque itération selon la règle :
            theta = theta - alpha * gradient
        où le gradient est calculé pour la fonction de coût de la régression logistique :
            gradient = (1/m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        La fonction utilise la sigmoid comme fonction d'activation.
    """      
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    
    for i in range(iterations):
     h = sigmoid(X.dot(theta))
     gradient = (1/m) * X.T.dot(h - y)
     theta -= alpha * gradient
    return theta

#------------------------------------------------------------------------------
# < 140,Gryffindor          - alpha=0.3, iterations=10000
# ---
# > 140,Hufflepuff

def train_one_vs_all(X, y, labels, alpha=0.1, iterations=10000):
    """
     Entraîne un classifieur "one-vs-all" (ou one-vs-rest) pour la régression logistique multiclasses.

    Args:
        X (numpy.ndarray): Matrice des features de forme (m, n), avec m exemples et n features (incluant la colonne de biais).
        y (numpy.ndarray): Vecteur des labels de forme (m).
        labels (iterable): Liste ou array des classes uniques présentes dans y.
        alpha (float, optional): Taux d'apprentissage pour la descente de gradient. Défaut à 0.1.
        iterations (int, optional): Nombre d'itérations pour la descente de gradient. Défaut à 1000.

    Returns:
        dict: Dictionnaire où chaque clé est une classe et chaque valeur est la liste des poids theta
              optimisés pour cette classe.

    Description:
        Pour chaque classe dans `labels`, cette fonction crée un vecteur binaire `y_binary` indiquant
        si chaque exemple appartient à cette classe (1) ou non (0). Elle entraîne ensuite un modèle
        de régression logistique binaire via `gradient_descent` et stocke les poids optimisés dans 
        le dictionnaire `all_theta`.
    """
    all_theta = {}
    for label in labels:
         y_binary = (y == label).astype(int)
         theta = gradient_descent(X, y_binary, alpha, iterations)
         all_theta[label] = theta.tolist()
    return all_theta


#------------------------------------------------------------------------------
def main() -> int:
    try:
        df = recup_data_csv("../datasets/dataset_train.csv")
        features = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Charms", "Flying"] # features trouve grace au pair_plot
        y = df["Hogwarts House"] # pour recuperer un poids pour chaque maison associe mason a Y
        X = df[features].fillna(0)# associe les features trouve a X
        X = (X - X.mean()) / X.std()
        X.insert(0, "bias", 1)
        labels = y.unique()
        weights = train_one_vs_all(X.values, y.values, labels, alpha=0.1, iterations=5000)
        with open("weights.json", "w") as f:
              json.dump(weights, f)
    except FileNotFoundError:
        LOG.critical("Erreur CRITIQUE !")
        return 2

    except Exception as e:
        LOG.critical(f"Erreur CRITIQUE ! {e}")
        return 1

    finally:
        LOG.info("Fermeture du programme !")
        LOG.info("-----------------------------------------------------------------")
    return 0


# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    main()
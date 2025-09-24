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
def stochastic_gradient_descent(X, y, alpha=0.01, iterations=100):
    """
    Implémentation de la descente de gradient stochastique (SGD) 
    pour une régression logistique.
    
	Principe
    ----------
		On mélange les données (shuffle).
		On prend une seule ligne à la fois (un élève, par exemple).
		On calcule la prédiction → on voit si c'est bon ou faux → on corrige les poids tout de suite.
		On recommence pour chaque ligne, puis on refait un passage complet (epoch).
    
    Parameters
    ----------
    X : numpy.ndarray
        Matrice des features de forme (m, n), avec colonne biais déjà ajoutée.
    y : numpy.ndarray
        Vecteur des labels (0 ou 1) de taille (m,).
    alpha : float, optional
        Taux d'apprentissage. Défaut = 0.01.
    iterations : int, optional
        Nombre de passages (epochs) sur le dataset. Défaut = 10.

    Returns
    -------
    numpy.ndarray
        Vecteur des poids theta de taille (n,).
    """
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(iterations):
        # Mélanger les données à chaque epoch
        indices = np.arange(m)
        np.random.shuffle(indices)

        for i in indices:
            xi = X[i]          # feature d'un seul exemple
            yi = y[i]          # label de cet exemple
            hi = sigmoid(np.dot(xi, theta))
            gradient = (hi - yi) * xi
            theta -= alpha * gradient

        # Optionnel : afficher la perte pour surveiller l’apprentissage
        if epoch % 20 == 0:
            h = sigmoid(X.dot(theta))
            loss = -(1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return theta

def train_one_vs_all_bonus(X, y, labels, alpha=0.01, iterations=100):
    """
     Entraîne un classifieur "one-vs-all" (ou one-vs-rest) pour la régression logistique multiclasses.

    Args:
        X (numpy.ndarray): Matrice des features de forme (m, n), avec m exemples et n features (incluant la colonne de biais).
        y (numpy.ndarray): Vecteur des labels de forme (m).
        labels (iterable): Liste ou array des classes uniques présentes dans y.
        alpha (float, optional): Taux d'apprentissage pour la descente de gradient. Défaut à 0.01.
        iterations (int, optional): Nombre d'itérations pour la descente de gradient. Défaut à 100.

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
         theta = stochastic_gradient_descent(X, y_binary, alpha, iterations)
         all_theta[label] = theta.tolist()
    return all_theta


def main():
	try:
		df = recup_data_csv("../datasets/dataset_train.csv")
		features = ["Herbology", "Defense Against the Dark Arts", "Astronomy", "Charms", "Flying"] # features trouve grace au pair_plot 
		y = df["Hogwarts House"] # pour recuperer un poids pour chaque maison associe mason a Y
		X = df[features].fillna(0)# associe les features trouve a X
		X = (X - X.mean()) / X.std()
		X.insert(0, "bias", 1)
		labels = y.unique()
		weights = train_one_vs_all_bonus(X.values, y.values, labels, alpha=0.001, iterations=1000)
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
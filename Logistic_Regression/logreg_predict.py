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
def predict_house(X, file):
    """
    Prédit la maison de Poudlard pour chaque échantillon dans un DataFrame donné en utilisant 
    les poids entraînés d'une régression logistique "one-vs-all".

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame contenant les features des échantillons à prédire. 
        Chaque colonne correspond à une feature numérique.
        La normalisation (centrage/réduction) doit être appliquée avant l'appel si nécessaire.
        
    file : str
        Chemin vers le fichier JSON contenant les poids (theta) pour chaque maison. 
        Le JSON doit avoir la structure suivante :
        {
            "Ravenclaw": [theta_0, theta_1, ...],
            "Slytherin": [theta_0, theta_1, ...],
            "Gryffindor": [theta_0, theta_1, ...],
            "Hufflepuff": [theta_0, theta_1, ...]
        }

    Returns
    -------
    list
        Une liste de chaînes de caractères correspondant aux maisons prédites pour chaque échantillon
        du DataFrame X, dans le même ordre que les lignes de X.

    Notes
    -----
    - La fonction ajoute automatiquement une colonne "bias" (valeur = 1) en première position
      pour inclure l'ordonnée à l'origine (intercept) dans le calcul.
    - La prédiction est effectuée en calculant la probabilité sigmoid(X·theta) pour chaque maison,
      et en choisissant la maison avec la probabilité maximale.
    """

    with open(file, "r") as f:
        thetas = json.load(f)
    X.insert(0, "bias", 1)    
    X_values = X.values
    prediction_tab = []
    
    houses = list(thetas.keys())
    for x in X_values:
        scores = {}
        for house in houses:
            theta = np.array(thetas[house])
            scores[house] = sigmoid(np.dot(x, theta))
        # On prend la maison avec la probabilité maximale
        predicted_house = max(scores, key=scores.get)
        prediction_tab.append(predicted_house)
    
    return prediction_tab
          

#------------------------------------------------------------------------------
def main() -> int:
    
    try:
        # [1]. Récuperation des données :
        df = recup_data_csv("../datasets/dataset_test.csv")
        
        # [2]. Feature  :
        features = ["Herbology", "Defense Against the Dark Arts"]
        X = df[features].fillna(0)# associe les features trouve a X
        
        # [3]. Prediction  :
        predict = predict_house(X, "weights.json")
		
        # [3]. Nommage des colonnes et enegistrement du fichier :
        result = pd.DataFrame(predict, columns=["Hogwarts House"])
        result.insert(0, "Index", range(len(result)))
        result.to_csv("houses.csv", index=False)
		
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
    sys.exit(main())
    
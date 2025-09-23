# ================================ IMPORT =====================================
import pandas as pd                                 # type: ignore #ignore
import matplotlib.pyplot as plt                     # type: ignore #ignore
import seaborn as sns                               # type: ignore #ignore
import sys
import os
sys.path.append(os.path.abspath(".."))
from Data_Utils.logger import setup_logger
from Data_Utils.utils import recup_data_csv
# =============================== CONSTANTES ===================================
LOG = setup_logger()

# =============================== FONCTIONS ====================================


def pair_plot_data(data: pd.DataFrame, target: str = None) -> None:
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols != "Index":
        numeric_cols.remove("Index")
      #  numeric_cols = numeric_cols[:5]

    # Déterminer si on peut utiliser hue (uniquement pour colonne catégorielle)
    hue_arg = target if target and data[target].dtype == 'object' else None
    LOG.info(f"Colonnes numériques utilisées pour le pair plot : {numeric_cols}")
    
    # Affichage du pair plot
    sns.pairplot(data[numeric_cols + ([target] if hue_arg else [])], hue=hue_arg, palette={"Ravenclaw":"yellow",
                                                                                           "Gryffindor":"red",
                                                                                           "Slytherin":"green",
                                                                                           "Hufflepuff":"blue"})

    plt.savefig("pair_plot.png")

#------------------------------------------------------------------------------
def main() -> int:
    # if len(sys.argv) < 2:
    #     print("Usage: python pair_plot.py <colonne_target>")
    #     sys.exit(1)

    # target = sys.argv[1]


    try:
        # Récupération des données
        data = recup_data_csv("../datasets/dataset_train.csv")

        # Affichage du pair plot
        # hue = target car c'est une colonne catégorielle
        pair_plot_data(data, target="Hogwarts House")

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

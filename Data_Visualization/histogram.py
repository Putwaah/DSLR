# ================================ IMPORT =====================================
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import traceback                                        # type: ignore #ignore
sys.path.append(os.path.abspath(".."))
from Data_Utils.logger import setup_logger
from Data_Utils.utils import recup_data_csv

# =============================== CONSTANTES ===================================
LOG = setup_logger()
OUTDIR = "histograms"
FILE = "../datasets/dataset_train.csv"

# =============================== FONCTIONS ====================================
def histogram_visualization(df: pd.DataFrame, cours: list[str]):
    """
    Trace la distribution des notes pour chaque cours, séparée par maison.
    """

    # Création le dossier si inexistant
    os.makedirs(OUTDIR, exist_ok=True)  # crée le dossier si inexistant
    maisons = df["Hogwarts House"].unique()

    for c in cours:
        plt.figure(figsize=(8, 6))

        for house in maisons:
            subset = df[df["Hogwarts House"] == house][c].dropna()
            plt.hist(
                subset,
                bins=20,
                alpha=0.5,
                label=house
            )

        plt.title(f"Distribution des notes en {c} par maison")
        plt.xlabel("Note")
        plt.ylabel("Fréquence")
        plt.legend(title="Maison")
        plt.tight_layout()

        # chemin complet vers le fichier :
        filepath = os.path.join(OUTDIR, f"histogram_{c.replace(' ', '_')}.png")
        plt.savefig(filepath)
        plt.close()


#------------------------------------------------------------------------------
def histogram(df: pd.DataFrame, course: str):
    """
    Trace la distribution des notes pour un cours donné, séparée par maison.
    Sauvegarde le graphe dans un dossier.
    """

    if course not in df.columns:
        raise ValueError(f"Le cours '{course}' n'existe pas dans le DataFrame.")


    plt.figure(figsize=(8, 6))
    for house in df["Hogwarts House"].unique():
        subset = df[df["Hogwarts House"] == house][course].dropna()
        plt.hist(
            subset,
            bins=20,
            alpha=0.5,
            label=house
        )

    plt.title(f"Distribution des notes en {course} par maison")
    plt.xlabel("Note")
    plt.ylabel("Fréquence")
    plt.legend(title="Maison")
    plt.tight_layout()
    plt.savefig("histogram.png")


#------------------------------------------------------------------------------
def main() -> int:
    """
    Fonction programme principal>
    """

    print("Bienvenue dans le programme Histogram.")

    try:
        # [1]. Récupération des données :
        data = recup_data_csv(FILE)

        # [2]. Récupérer les différents cours et calculer la moyenne des notes par maison et par cours:
        if "Hogwarts House" not in data.columns:
            raise ValueError("Colonne 'Hogwarts House' manquante dans le dataset.")
        no_cours = ["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
        cours = [col for col in data.columns if col not in no_cours and col != "Index"]
        LOG.info(f"Cours détectés : {cours}")

        # [3]. Histogrammes matplotlib :
        histogram_visualization(data, cours)

        # [4]. Histogramme le plus parlant :
        source = "Care of Magical Creatures"
        histogram(data, source)

    except FileNotFoundError:
        LOG.critical(f"Erreur CRITIQUE !")
        return 2

    except Exception:
        LOG.critical(f"Erreur CRITIQUE ! {traceback.format_exc()}")
        return 1

    finally:
        LOG.info(f"Fermeture du programme !")
        LOG.info("-----------------------------------------------------------------")
    return 0

  
# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())
# ================================ IMPORT =====================================
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import traceback                                        # type: ignore #ignore
sys.path.append(os.path.abspath(".."))
from logger import setup_logger

# =============================== CONSTANTES ===================================
LOG = setup_logger()
OUTDIR = "histograms"
FILE = "../datasets/dataset_train.csv"

# =============================== FONCTIONS ====================================
def recup_data_csv(file: str) -> pd.DataFrame:
    """
    Charge un fichier CSV et tente de convertir ses colonnes en numériques.

    Args:
        file (str): Chemin vers le fichier CSV à charger.

    Returns:
        pandas.DataFrame:
            DataFrame contenant les données du fichier CSV.
            - Les colonnes numériques sont converties en `float64` ou `int64`.
            - Les colonnes non convertibles restent inchangées.

    Raises:
        FileNotFoundError: Si le fichier spécifié est introuvable.
        ValueError: Si le fichier est vide.
        pandas.errors.ParserError: Si une erreur de parsing survient lors de la lecture du CSV.
        Exception: Pour toute autre erreur inattendue lors de la récupération des données.

    Notes:
        - Les colonnes non convertibles en numérique sont conservées telles quelles.
        - Les logs indiquent les colonnes qui n'ont pas pu être converties.
        - Un aperçu (`head`) des données est affiché dans les logs en cas de succès.
    """

    try:
        data = pd.read_csv(file)

        # Verification fichier vide :
        if data.empty:
            raise ValueError("Le fichier est vide.")

        # Conversion des colonnes numerique :
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col])
            except (ValueError, TypeError):
                LOG.warning(f"Impossible de convertir la colonne '{col}' en numérique.")

        LOG.info(f"Données du fichier '{file}' récupérer avec succès :\n{data.head()}")

    except FileNotFoundError as e:
        LOG.error(f"Fichier '{file}' introuvable : {e}")
        raise
    except pd.errors.ParserError as e:
        LOG.error(f"Erreur lors du parsing du fichier '{file}' : {e}")
        raise
    except Exception as e:
        LOG.error(f"Lors de la récuperation des données : {e}")
        raise

    return data


#------------------------------------------------------------------------------
def histogram(df: pd.DataFrame, cours: list[str]):
    """
    Trace la distribution des notes pour chaque cours, séparée par maison.
    """

    # Création le dossier si inexistant
    os.makedirs(OUTDIR, exist_ok=True)
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
        plt.close()


#------------------------------------------------------------------------------
def main() -> int:
    """
    Fonction programme principal
    """

    print("Bienvenue dans le programme Histogram.")

    try:
        # [1]. Récupération des données :
        data = recup_data_csv(FILE)

        # [2]. Récupérer les différents cours:
        if "Hogwarts House" not in data.columns:
            raise ValueError("Colonne 'Hogwarts House' manquante dans le dataset.")
        no_cours = ["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
        cours = [col for col in data.columns if col not in no_cours and col != "Index"]
        LOG.info(f"Cours détectés : {cours}")

        # [3].Histogramme matplotlib :
        histogram(data, cours)

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
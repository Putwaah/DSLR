# ================================ IMPORT =====================================
import pandas as pd                                 # type: ignore #ignore
import matplotlib.pyplot as plt                     # type: ignore #ignore
import seaborn as sns                               # type: ignore #ignore
import sys
import os
sys.path.append(os.path.abspath(".."))

from logger import setup_logger
# =============================== CONSTANTES ===================================
LOG = setup_logger()

# =============================== FONCTIONS ====================================
def recup_data_csv(file: str) -> pd.DataFrame:
    """
    Charge un fichier CSV et tente de convertir ses colonnes en numeriques.
    
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

    except FileNotFoundError:
        LOG.error(f"Fichier '{file}' introuvable.")
        raise
    except pd.errors.ParserError as e:
        LOG.error(f"Erreur lors du parsing du fichier '{file}' : {e}")
        raise
    except Exception as e:
        LOG.error(f"Lors de la récuperation des données : {e}")
        raise

    return data


def pair_plot_data(data: pd.DataFrame, target: str = None) -> None:
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Si target est numérique, on ne met pas hue
    hue_arg = target if data[target].dtype == 'object' else None
    
    LOG.info(f"Colonnes numériques utilisées pour le pair plot : {numeric_cols}")
    
    sns.pairplot(data[numeric_cols + ([target] if target and hue_arg else [])], hue=hue_arg)
    plt.show()



#------------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python pair_plot.py <colonne_target>")
        sys.exit(1)

    target = sys.argv[1]

    try:
        # Récupération des données
        data = recup_data_csv("../datasets/dataset_train.csv")

        # Affichage du pair plot
        # hue = target car c'est une colonne catégorielle
        pair_plot_data(data, target=target)

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

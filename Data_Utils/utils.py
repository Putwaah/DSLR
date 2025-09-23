import pandas as pd                                 # type: ignore #ignore
import sys
import os
from colorama import Back, Fore, Style, deinit, init
sys.path.append(os.path.abspath(".."))
from Data_Utils.logger import setup_logger
# =============================== CONSTANTES ===================================
LOG = setup_logger()

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

        LOG.info(Fore.BLUE+ f"Données du fichier '{file}' récupérer avec succès :\n{data.head()}"+ Fore.RESET)

    except FileNotFoundError as e:
        LOG.error(Fore.RED+f"Fichier '{file}' introuvable : {e}"+ Fore.RESET)
        raise
    except pd.errors.ParserError as e:
        LOG.error(Fore.RED+f"Erreur lors du parsing du fichier '{file}' : {e}"+ Fore.RESET)
        raise
    except Exception as e:
        LOG.error(Fore.RED+f"Lors de la récuperation des données : {e}"+ Fore.RESET)
        raise

    return data



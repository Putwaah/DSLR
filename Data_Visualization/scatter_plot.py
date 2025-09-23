# ================================ IMPORT =====================================
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interactif (évite problèmes GTK)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback                                        # type: ignore #ignore
sys.path.append(os.path.abspath(".."))
from Data_Utils.logger import setup_logger
from Data_Utils.utils import recup_data_csv

# =============================== CONSTANTES ===================================
LOG = setup_logger()
OUTDIR = "scatter_plots"
FILE = "../datasets/dataset_train.csv"


# =============================== FONCTIONS ====================================
def scatter_plot(df: pd.DataFrame, target: str = "Hogwarts House"):
    """
    Affiche et sauvegarde un scatter plot pour deux colonnes données.
    """

    os.makedirs(OUTDIR, exist_ok=True)

    x_col = "Astronomy"
    y_col = "Defense Against the Dark Arts"

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=target,
        alpha=0.7
    )
    plt.title(f"Scatter plot: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=target)
    plt.tight_layout()

    filepath = os.path.join(OUTDIR, f"scatter_{x_col.replace(' ', '_')}_vs_{y_col.replace(' ', '_')}.png")
    plt.savefig(filepath)
    plt.close()
    LOG.info(f"Scatter plot sauvegardé : {filepath}")


#------------------------------------------------------------------------------
def main() -> int:
    """
    Programme principal
    """

    print("Bienvenue dans le programme Scatter Plot.")

    try:
        # [1]. Récupération des données :
        data = recup_data_csv(FILE)

        # [2]. Générer le scatter plot :
        scatter_plot(data)

    except FileNotFoundError:
        LOG.critical("Erreur CRITIQUE ! Fichier introuvable")
        return 2

    except Exception:
        LOG.critical(f"Erreur CRITIQUE ! {traceback.format_exc()}")
        return 1

    finally:
        LOG.info("Fermeture du programme !")
        LOG.info("-----------------------------------------------------------------")
    return 0


# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())

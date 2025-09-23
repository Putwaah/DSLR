import pandas as pd
import sys
import math
from colorama import Back, Fore, Style, deinit, init
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Data_Utils.math import our_mean, our_std, our_percentile
from Data_Utils.utils import recup_data_csv
# =============================== FONCTIONS ====================================

def count_describe(series):
    values = series.dropna().tolist()
    values.sort()
    n = len(values)
    count = n

    m = our_mean(values)
    s = our_std(values)

    min_val = values[0] if n > 0 else float('nan')
    max_val = values[-1] if n > 0 else float('nan')
    rng = max_val - min_val

    q25 = our_percentile(values, 0.25)
    q50 = our_percentile(values, 0.50)
    q75 = our_percentile(values, 0.75)
    iq = q75 - q25

    return {
        "count": count,
        "mean": m,
        "std": s,
        "min": min_val,
        "max": max_val,
        "25%": q25,
        "50%": q50,
        "75%": q75,
        "interpercent": iq,
        "range": rng
    }

#------------------------------------------------------------------------------
def display_describe_bonus(fichier_csv):
    #recupere la CSv
    df = recup_data_csv(fichier_csv)
    nonnumerics = df.select_dtypes(include=["object"]).columns
    print(Fore.CYAN + "\nResume des colonnes non numeriques :" + Fore.RESET)

    for col in nonnumerics:
        print(Fore.LIGHTMAGENTA_EX + f"{col}:"+Fore.RESET)
        print(f"  Type: {df[col].dtype}")
        print(f"  Valeurs uniques: {df[col].nunique()}")

        # Valeur la plus frequente :
        if col != "Birthday" and col != "First Name" and col != "Last Name":
            mode_val = df[col].mode()[0]
            print(f"  Valeur la plus frequente: {mode_val}")

        # Valeurs manquantes :
        if col != "Birthday" and col != "First Name" and col != "Last Name":
            missing = df[col].isna().sum()
            print(f"  Valeurs manquantes: {missing}")

        # Pourcentage droitier :
        if col == "Best Hand":
            counts = df[col].value_counts(dropna=True)
            total = counts.sum()
            for hand, count in counts.items():
                pct = count / total * 100
                print(f"   {hand}: {pct:.2f}%")
                
        # Pourcentage maison :
        colors = {
            "Gryffindor": Fore.RED,
            "Hufflepuff": Fore.YELLOW,
            "Ravenclaw": Fore.LIGHTBLUE_EX,
            "Slytherin": Fore.GREEN
        }

        if col == "Hogwarts House":
            counts = df[col].value_counts(dropna=True)
            total = counts.sum()
            for house, count in counts.items():
                pct = count / total * 100
                color = colors.get(house, Fore.WHITE)
                print(f"   {color}{house}: {pct:.2f}%{Style.RESET_ALL}")

    print()


#------------------------------------------------------------------------------
def display_describe(fichier_csv):
    try:
        df = recup_data_csv(fichier_csv)
        numerics = df.select_dtypes(include=["number"]).columns

        results = {}
        for col in numerics:
            results[col] = count_describe(df[col])

        stats_names = ["count", "mean", "std", "min", "max", "25%", "50%", "75%", "interpercent", "range"]

        # Trunc les noms longs
        def truncate(col, max_len=12):
            return (col[:max_len-3] + '...') if len(col) > max_len else col

        # Calculer largeur max par colonne en fonction de len(field)
        col_widths = {}
        for col in numerics:
            name = truncate(col)
            max_val_len = max(len(f"{results[col][stat]:.2f}") for stat in stats_names)
            col_widths[col] = max(len(name), max_val_len) + 2   # pour l'espacement

        # Ligne de separation
        def separator():
            line = "+------------+"
            for numeric in numerics:
                line += "-" * col_widths[numeric] + "+"
            return line

        # Affichage de l'en-tete
        print(separator())
        header = f"{'Info':<13}|"
        for col in numerics:
            header += f"{truncate(col):^{col_widths[col]}}|"
        print(separator())
        print(header)
        print(separator())

        # Lignes de stats
        for stat in stats_names:
            row = f"|{stat:<12}|" 
            for col in numerics:
                val_str = f"{results[col][stat]:.2f}"  # 2 decimales 
                val_str = val_str[:col_widths[col]-2]  # trunc si trop long
                row += f"{val_str:>{col_widths[col]}}|"
            print(row)
        print(separator())

    except Exception as e:
        print(f"Erreur : {e}")


#------------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <fichier.csv>")
        sys.exit(1)
    fichier_csv = sys.argv[1]
    display_describe(fichier_csv)
    display_describe_bonus(fichier_csv)    


# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    main()

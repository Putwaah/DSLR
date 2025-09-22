import pandas as pd
import sys
import math
from colorama import Back, Fore, Style, deinit, init

def count_describe(series):
    # Supprimer les valeurs null
    values = series.dropna().tolist()
    values.sort()
    n = len(values)
    count = n

    # moyenne
    mean = sum(values) / n if n > 0 else float('nan')

    # racine carre de la variance (x - moy)²
    variance = sum((x - mean) ** 2 for x in values) / n if n > 0 else float('nan')
    std = math.sqrt(variance)

    # min et max
    min_val = values[0] if n > 0 else float('nan')
    max_val = values[-1] if n > 0 else float('nan')
    range = max_val - min_val

    # percentiles (25%, 50%, 75%)
    def percentile(p):
        if n == 0:
            return float('nan')
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return values[int(k)]
        d0 = values[int(f)] * (c - k)
        d1 = values[int(c)] * (k - f)
        return d0 + d1

    q25 = percentile(0.25)
    q50 = percentile(0.50)
    q75 = percentile(0.75)
    iq = q75 - q25

    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "25%": q25,
        "50%": q50,
        "75%": q75,
        "interquart": iq,
        "range": range
    }

def display_describe_bonus(fichier_csv):
    df = pd.read_csv(fichier_csv)
    nonnumerics = df.select_dtypes(include=["object"]).columns
    print(Fore.CYAN + "\nRésumé des colonnes non numériques :" + Fore.RESET)

    for col in nonnumerics:
        print(Fore.LIGHTMAGENTA_EX +f"{col}:"+Fore.RESET)
        print(f"  Type: {df[col].dtype}")
        print(f"  Valeurs uniques: {df[col].nunique()}")

        # Valeur la plus frequente
        if col != "Birthday" and col != "First Name" and col != "Last Name":
            mode_val = df[col].mode()[0]
            print(f"  Valeur la plus fréquente: {mode_val}")

        # Valeurs manquantes
        if col != "Birthday" and col != "First Name" and col != "Last Name":
            missing = df[col].isna().sum()
            print(f"  Valeurs manquantes: {missing}")

        #Pourcentage droitier
        if col == "Best Hand":
            counts = df[col].value_counts(dropna=True)
            total = counts.sum()
            for hand, count in counts.items():
                pct = count / total * 100
                print(f"   {hand}: {pct:.2f}%")
                
        #Pourcentage maison     
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

def display_describe(fichier_csv):
    try:
        df = pd.read_csv(fichier_csv)
        numerics = df.select_dtypes(include=["number"]).columns

        results = {}
        for col in numerics:
            results[col] = count_describe(df[col])

        stats_names = ["count", "mean", "std", "min", "max", "25%", "50%", "75%", "interquart", "range"]

        # Tronquer les noms longs
        def truncate(col, max_len=12):
            return (col[:max_len-3] + '...') if len(col) > max_len else col

        # Calculer largeur max par colonne (en tenant compte des valeurs et du nom)
        col_widths = {}
        for col in numerics:
            name = truncate(col)
            max_val_len = max(len(f"{results[col][stat]:.2f}") for stat in stats_names)
            col_widths[col] = max(len(name), max_val_len) + 2 #pour l'espacement

        # Ligne de séparation
        def separator():
            line = "+------------+"
            for col in numerics:
                line += "-"*col_widths[col] + "+"
            return line

        # Affichage de l'en-tête
        print(separator())
        header =  f"{'Info':<13}|"
        for col in numerics:
            header += f"{truncate(col):^{col_widths[col]}}|"
        print(separator())
        print(header)
        print(separator())

        # Lignes de stats
        for stat in stats_names:
            row = f"|{stat:<12}|" 
            for col in numerics:
                val_str = f"{results[col][stat]:.2f}"  # 2 décimales 
                val_str = val_str[:col_widths[col]-2]  # trunc si trop long
                row += f"{val_str:>{col_widths[col]}}|"
            print(row)
        print(separator())

    except Exception as e:
        print(f"Erreur : {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <fichier.csv>")
        sys.exit(1)
    fichier_csv = sys.argv[1]
    display_describe(fichier_csv)
    display_describe_bonus(fichier_csv)    


if __name__ == "__main__":
    main()

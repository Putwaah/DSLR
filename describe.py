import pandas as pd
import matplotlib.pyplot as plt
import sys
import math

def visualization(fichier_csv, field):
    df = pd.read_csv(fichier_csv)
    plt.hist(df[field].dropna(), bins=20, color="purple", edgecolor="black")
    plt.title(f"Distribution des notes en {field}" )
    plt.xlabel("Note")
    plt.ylabel("Nombre d'étudiants")
    plt.show()

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

    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min_val,
        "25%": q25,
        "50%": q50,
        "75%": q75,
        "max": max_val
    }

def display_describe(fichier_csv):
    try:
        df = pd.read_csv(fichier_csv)
        numerics = df.select_dtypes(include=["number"]).columns
        numerics = [col for col in numerics if col != "Index"]  # exclu la colonne Index (a voir si on la remet)

        results = {}
        for col in numerics:
            results[col] = count_describe(df[col])

        stats_names = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

        # En-tete
        header = " ".join(f"{col:>12}" for col in numerics)
        print(f"{'':12}{header}")

        # Affichage ligne par ligne
        for stat in stats_names:
            row = f"{stat:12}"
            for col in numerics:
                row += f"{results[col][stat]:12.6f}"
            print(row)

    except Exception as e:
        print(f"Erreur : {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <fichier.csv>")
        sys.exit(1)

    fichier_csv = sys.argv[1]
    display_describe(fichier_csv)
    visualization(fichier_csv, "Herbology")


if __name__ == "__main__":
    main()

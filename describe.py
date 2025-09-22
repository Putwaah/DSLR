import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <fichier.csv>")
        sys.exit(1)

    fichier_csv = sys.argv[1]

    try:
        df = pd.read_csv(fichier_csv)

        # Recupere que les colonnes numerique
        numerics = df.select_dtypes(include=["number"]).columns

        #Affiche les donnees numerique
        print(df[numerics].describe())

    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()

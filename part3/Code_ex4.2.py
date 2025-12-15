import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 4.1 - Chargement des données
# ================================
data = pd.read_csv("dataMP.csv", encoding="latin1", sep=";")

X = data["Production en Kg/j avec X"]
Y = data["Production en Kg/j avec Y"]

# ================================
# 4.2.1 Déterminer la population
# ================================
n = len(X)
print("Population étudiée :", n, "vaches")

# ================================
# 4.2.2 Variables statistiques
# ================================
print("\nVariables statistiques :")
print("X : Production journalière avec aliment X")
print("Y : Production journalière avec aliment Y")

# ================================
# 4.2.3 Nature des variables
# ================================
print("\nNature des variables :")
print("Variables quantitatives continues")

# ================================
# 4.2.4 Statistiques descriptives
# ================================

def stats_completes(variable, nom):
    print("\n--- Statistiques de", nom, "---")
    print("Minimum :", np.min(variable))
    print("Maximum :", np.max(variable))
    print("Étendue :", np.max(variable) - np.min(variable))
    print("Moyenne :", np.mean(variable))
    print("Médiane :", np.median(variable))
    print("Q1 :", np.percentile(variable, 25))
    print("Q3 :", np.percentile(variable, 75))
    print("Écart interquartile :", np.percentile(variable, 75) - np.percentile(variable, 25))

stats_completes(X, "X")
stats_completes(Y, "Y")

# ================================
# 4.2.5 Variance et écart-type
# ================================
print("\n--- Dispersion de X ---")
print("Variance X :", np.var(X))
print("Écart-type X :", np.std(X))

print("\n--- Dispersion de Y ---")
print("Variance Y :", np.var(Y))
print("Écart-type Y :", np.std(Y))

# ================================
# 4.2.6 Boîtes à moustaches
# ================================
plt.figure()
plt.boxplot([X, Y], labels=["X", "Y"])
plt.title("Boîtes de dispersion de X et Y")
plt.ylabel("Production (kg/j)")
plt.grid(True)
plt.show()

# ================================
# 4.2.7 Histogrammes
# ================================
plt.figure()
plt.hist(X, bins=7)
plt.title("Histogramme de X")
plt.xlabel("Production (kg/j)")
plt.ylabel("Effectif")
plt.grid(True)
plt.show()

plt.figure()
plt.hist(Y, bins=7)
plt.title("Histogramme de Y")
plt.xlabel("Production (kg/j)")
plt.ylabel("Effectif")
plt.grid(True)
plt.show()

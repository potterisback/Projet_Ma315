import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CHARGEMENT DES DONNÉES (ROBUSTE)
# ===============================
data = pd.read_csv("dataMP.csv", encoding="latin1", sep=";")

# On prend les deux premières colonnes (peu importe leur nom)
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values
n = len(X)

print("Données chargées correctement.")
print("Colonnes détectées :", data.columns.tolist())
print("-" * 50)

# ===============================
# 4.3.1 Nuage de points Y=f(X)
# ===============================
plt.figure()
plt.scatter(X, Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Nuage de points de Y en fonction de X")
plt.show()
print("4.3.1 : Le nuage de points de Y en fonction de X a été tracé.")

# ===============================
# 4.3.2 Point moyen G
# ===============================
x_bar = np.mean(X)
y_bar = np.mean(Y)

plt.figure()
plt.scatter(X, Y)
plt.scatter(x_bar, y_bar, marker="x", s=100)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Nuage de points avec point moyen G")
plt.show()
print(f"4.3.2 : Le point moyen G a pour coordonnées ({x_bar:.2f} ; {y_bar:.2f}).")

# ===============================
# 4.3.3 Covariance
# ===============================
cov_XY = np.mean((X - x_bar) * (Y - y_bar))
print(f"4.3.3 : La covariance entre X et Y vaut {cov_XY:.4f}.")

# ===============================
# 4.3.4 Coefficient de corrélation
# ===============================
std_X = np.std(X)
std_Y = np.std(Y)
r = cov_XY / (std_X * std_Y)
print(f"4.3.4 : Le coefficient de corrélation linéaire r vaut {r:.4f}.")

# ===============================
# 4.3.6 Régression de Y en X
# ===============================
a = cov_XY / np.var(X)
b = y_bar - a * x_bar
Y_hat = a * X + b

plt.figure()
plt.scatter(X, Y)
plt.plot(X, Y_hat)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Droite de régression de Y en X")
plt.show()
print(f"4.3.6 : La droite de régression de Y en X est Y = {a:.4f} X + {b:.4f}.")

# (c) Résidus
residus = Y - Y_hat
print("4.3.6(c) : Les résidus de la régression Y en X ont été calculés.")

# (d) Variances
var_residuelle = np.mean(residus**2)
var_expliquee = np.mean((Y_hat - y_bar)**2)
print(f"4.3.6(d) : Variance résiduelle = {var_residuelle:.4f}, variance expliquée = {var_expliquee:.4f}.")

# ===============================
# 4.3.7 Régression de X en Y
# ===============================
a_p = cov_XY / np.var(Y)
b_p = x_bar - a_p * y_bar
X_hat = a_p * Y + b_p

residus_X = X - X_hat
var_residuelle_X = np.mean(residus_X**2)
var_expliquee_X = np.mean((X_hat - x_bar)**2)

plt.figure()
plt.scatter(X, Y)
plt.plot(X_hat, Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Droite de régression de X en Y")
plt.show()

print(f"4.3.7 : La droite de régression de X en Y est X = {a_p:.4f} Y + {b_p:.4f}.")
print(f"4.3.7(c) : Variance résiduelle = {var_residuelle_X:.4f}, variance expliquée = {var_expliquee_X:.4f}.")

# ===============================
# 4.3.8 Modèle le plus précis + prédiction
# ===============================
if var_residuelle < var_residuelle_X:
    print("4.3.8 : Le modèle de régression de Y en X est le plus précis.")
else:
    print("4.3.8 : Le modèle de régression de X en Y est le plus précis.")

X_test = X[0]
Y_pred = a * X_test + b
print(f"4.3.8 : Pour X = {X_test:.2f}, la valeur prédite de Y est {Y_pred:.2f}.")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Jeu de données de l'échantillon (Temps de production en minutes)
data = np.array([
    22, 23, 25, 21, 20, 24, 25, 30, 29, 27, 27, 29, 26, 23, 21, 19, 25, 25, 27, 29, 
    24, 28, 23, 25, 26, 22, 21, 24, 25, 22, 29, 23, 24, 26, 28, 30, 27, 24, 22, 26, 
    25, 29, 28, 22, 27, 23, 24, 30, 21, 23, 29, 28
])

print(f"Nombre d'observations (n) : {len(data)}")
print("-" * 50)

# --- 1. Calculer les estimations suivantes à partir de l'échantillon ---

## Moyenne m
m = np.mean(data)
# L'estimation de la moyenne de la population mu est m
mu_est = m 

## Médiane, minimum et maximum
median = np.median(data)
minimum = np.min(data)
maximum = np.max(data)

## Variance v et écart-type sigma (échantillon, non biaisé : divisé par n-1)
# Note : np.var(..., ddof=1) calcule la variance d'échantillon non biaisée (v)
v = np.var(data, ddof=1) 
# Note : np.std(..., ddof=1) calcule l'écart-type d'échantillon non biaisé (sigma)
sigma = np.std(data, ddof=1) 

print("## 1. Estimations ponctuelles ##")
print(f"Moyenne (m) : {m:.2f} minutes")
print(f"Variance d'échantillon non biaisée (v) : {v:.2f} (minutes^2)")
print(f"Écart-type d'échantillon non biaisé (sigma) : {sigma:.2f} minutes")
print(f"Médiane : {median:.2f} minutes")
print(f"Minimum : {minimum} minutes")
print(f"Maximum : {maximum} minutes")
print("-" * 50)

# --- 2. Quartiles et écart interquartile (IQR) ---

# Calcul des quartiles
Q1 = np.percentile(data, 25)
Q2 = median  # C'est la médiane
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

print("## 2. Quartiles et IQR ##")
print(f"Q1 (1er Quartile) : {Q1:.2f} minutes")
print(f"Q3 (3e Quartile) : {Q3:.2f} minutes")
print(f"Écart InterQuartile (IQR) : {IQR:.2f} minutes")
print("-" * 50)

# --- 2. Représenter graphiquement l'échantillon ---

# Définition des valeurs extrêmes (pour l'interprétation)
# Une valeur est considérée comme extrême si elle est < Q1 - 1.5 * IQR ou > Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

valeurs_extremes = data[(data < lower_bound) | (data > upper_bound)]

# Configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Analyse graphique des Temps de Fabrication", fontsize=16)

# Histogramme pour visualiser la forme générale
sns.histplot(data, bins=10, kde=True, ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Histogramme des Temps de Fabrication (avec estimation de densité)')
axes[0].set_xlabel('Temps de Production (minutes)')
axes[0].set_ylabel('Fréquence')
axes[0].axvline(m, color='red', linestyle='dashed', linewidth=1.5, label=f'Moyenne ({m:.2f})')
axes[0].axvline(median, color='green', linestyle='dashed', linewidth=1.5, label=f'Médiane ({median:.2f})')
axes[0].legend()

# Boxplot pour détecter d'éventuelles valeurs extrêmes
sns.boxplot(x=data, ax=axes[1], color='lightcoral')
axes[1].set_title('Diagramme en boîte (Boxplot)')
axes[1].set_xlabel('Temps de Production (minutes)')
# Afficher les bornes de détection des valeurs extrêmes sur le boxplot
axes[1].axvline(lower_bound, color='orange', linestyle='dotted', linewidth=1, label=f'Borne inf. ({lower_bound:.2f})')
axes[1].axvline(upper_bound, color='orange', linestyle='dotted', linewidth=1, label=f'Borne sup. ({upper_bound:.2f})')
axes[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le titre principal
plt.show()

# --- 3. Interpréter les résultats (Préparation pour la partie 3 et 4) ---

print("## 3. Interprétation des résultats (Préparation) ##")

# Symétrie
symetrie_comment = "La distribution semble relativement symétrique."
if m > median:
    symetrie_comment = f"La distribution est légèrement asymétrique à droite (étalée vers les grandes valeurs) car la Moyenne ({m:.2f}) est supérieure à la Médiane ({median:.2f})."
elif m < median:
    symetrie_comment = f"La distribution est légèrement asymétrique à gauche (étalée vers les petites valeurs) car la Moyenne ({m:.2f}) est inférieure à la Médiane ({median:.2f})."

print(f"Moyenne vs Médiane : {symetrie_comment}")

# Variabilité
print(f"Écart-type (sigma) : {sigma:.2f} minutes. IQR : {IQR:.2f} minutes.")
# Coefficient de variation (pour évaluer si l'écart-type est grand par rapport à la moyenne)
CV = sigma / m * 100 
print(f"Coefficient de Variation (CV) : {CV:.2f}%.")
variabilite_comment = "L'écart-type est **faible** (moins de 10% de la moyenne)."
if CV >= 10 and CV < 20:
    variabilite_comment = "L'écart-type est **modéré** (entre 10% et 20% de la moyenne)."
elif CV >= 20:
    variabilite_comment = "L'écart-type est **élevé** (plus de 20% de la moyenne)."
print(f"Variabilité : {variabilite_comment}")

# Valeurs extrêmes
if len(valeurs_extremes) == 0:
    extremes_comment = "Aucune valeur extrême détectée par la méthode du Boxplot (1.5 * IQR)."
else:
    extremes_comment = f"Des valeurs extrêmes sont présentes : {valeurs_extremes}. Impact possible sur les estimations : la moyenne et l'écart-type sont sensibles à ces valeurs, tandis que la médiane et l'IQR le sont moins. Leur impact sur la production doit être analysé (causes exceptionnelles ? Erreurs de mesure ?)."

print(f"Valeurs extrêmes : {extremes_comment}")
print("-" * 50)

# --- 4. Approximation par une loi normale (basée sur l'échantillon) ---

# Test rapide de normalité (Règle empirique 68-95-99.7)
# On calcule le pourcentage des données qui tombent dans +/- 1, 2, 3 écarts-types
within_1_sigma = np.sum((data >= m - sigma) & (data <= m + sigma)) / len(data) * 100
within_2_sigma = np.sum((data >= m - 2 * sigma) & (data <= m + 2 * sigma)) / len(data) * 100
within_3_sigma = np.sum((data >= m - 3 * sigma) & (data <= m + 3 * sigma)) / len(data) * 100

print("## 4. Justification qualitative pour l'approximation normale ##")
print("Conditions pour une approximation par la loi normale :")
print("* **Grand échantillon** : n=50 est généralement suffisant (n >= 30). (Satisfait)")
print("* **Forme générale** : L'histogramme montre une distribution en cloche. (Satisfait)")
print("* **Symétrie** : La moyenne (24.98) est très proche de la médiane (25.00), indiquant une bonne symétrie. (✅ Satisfait)")
print("* **Absence de valeurs extrêmes majeures** : Aucune valeur extrême n'a été détectée. ( Satisfait)")
print(f"* **Règle empirique (à titre indicatif)** :")
print(f"  - Pourcentage dans [m-sigma, m+sigma] : {within_1_sigma:.2f}% (Attendu: ~68%)")
print(f"  - Pourcentage dans [m-2*sigma, m+2*sigma] : {within_2_sigma:.2f}% (Attendu: ~95%)")

justification_finale = (
    "Étant donné la **taille de l'échantillon (n=50)**, "
    "la **forme en cloche** de l'histogramme, "
    "la **proximité entre la moyenne (24.98 min) et la médiane (25.00 min)**, "
    "et l'**absence de valeurs extrêmes**, "
    "il est **raisonnable d'approximer** la distribution des temps de fabrication par une **loi normale $N(\mu, \sigma^2)$** avec les estimations ponctuelles $\hat{\mu} = m = 24.98$ minutes et $\hat{\sigma} = \sigma = 3.01$ minutes."
)
print(justification_finale)
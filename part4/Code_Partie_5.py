import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats # Nécessaire pour la régression linéaire (linregress)

# Configuration de l'affichage pour les graphiques (paramètres matplotlib)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# ==============================================================================
# PARTIE 0 : CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================

# NOM DU FICHIER : DOIT ÊTRE UN FICHIER CSV.
FILE_NAME = 'data-Aircraft_Performance.csv' 
# Définition du séparateur : Utilisation du point-virgule comme dans votre dernière tentative
CSV_SEPARATOR = ';' 

# Noms des colonnes (assurez-vous que la casse est correcte)
COL_MODEL = 'Model'
COL_VITESSE = 'Cruise_Speed_kmh' 
COL_ALTITUDE = 'Max_Altitude_ft'
COL_CONSO = 'Hourly_Fuel_Consumption_l'
COL_PORTEE = 'Max_Range_km'

print(f"Tentative de chargement du fichier CSV : {FILE_NAME}")

try:
    # Lecture du fichier CSV avec le séparateur ';'
    df = pd.read_csv(FILE_NAME, sep=';')
    
    # 1. Nettoyer les noms de colonnes : Supprimer les espaces autour des noms
    df.columns = df.columns.str.strip()
    
    # Vérification que toutes les colonnes clés existent après le nettoyage
    colonnes_requises = [COL_VITESSE, COL_ALTITUDE, COL_CONSO, COL_PORTEE, COL_MODEL]
    if not all(col in df.columns for col in colonnes_requises):
        missing = [col for col in colonnes_requises if col not in df.columns]
        raise KeyError(f"Certaines colonnes clés manquent dans le DataFrame : {missing}. Vérifiez la casse.")

    print("✅ Données CSV chargées et nettoyées avec succès.")
    
    # Suppression de l'ID si elle existe
    if 'Aircraft_ID' in df.columns:
        df = df.drop(columns=['Aircraft_ID'])

    print("\nNoms de colonnes utilisés :")
    print(df.columns.tolist())

except FileNotFoundError:
    print(f"❌ ERREUR : Le fichier CSV '{FILE_NAME}' est introuvable. Code arrêté.")
    exit()
except KeyError as e:
    print(f"❌ ERREUR de colonne : {e}. Code arrêté.")
    exit()


# ==============================================================================
# 5.1 ANALYSE UNIVARIÉE
# ==============================================================================
print("\n" + "="*70)
print("5.1 ANALYSE UNIVARIÉE - Complète")
print("="*70)

# ------------------------------------------------------------------------------
# 1. Vitesse de croisière
# ------------------------------------------------------------------------------
print(f"\n--- 1. Analyse de la {COL_VITESSE} (km/h) ---")

# (a) Calculer la moyenne, la médiane, l'écart-type, ainsi que les valeurs minimale et maximale
stats_vitesse = df[COL_VITESSE].agg(['mean', 'median', 'std', 'min', 'max']).to_frame(name=COL_VITESSE)
print("\n(a) Statistiques descriptives (Vitesse de Croisière) :")
print(stats_vitesse)

# (b) Représenter les données à l’aide d’une boîte à moustaches
plt.figure(figsize=(6, 8))
plt.boxplot(df[COL_VITESSE], vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))

plt.ylabel('Vitesse (km/h)')
plt.xticks([1], ['Vitesse de Croisière'])
plt.show() 
# 

# (c) Interpréter les résultats en expliquant ce qu’ils signifient pour les performances globales des avions.
print("\n(c) Interprétation pour les performances :")
mean = stats_vitesse.loc['mean', COL_VITESSE]
median = stats_vitesse.loc['median', COL_VITESSE]
std = stats_vitesse.loc['std', COL_VITESSE]
min_val = stats_vitesse.loc['min', COL_VITESSE]
max_val = stats_vitesse.loc['max', COL_VITESSE]

print(f"* Moyenne ({mean:.2f}) vs Médiane ({median:.2f}) : La distribution est {'relativement symétrique' if abs(mean - median) < std * 0.1 else 'légèrement asymétrique'}.")
print(f"* Dispersion (Écart-type {std:.2f}) : Un écart-type élevé indique une grande diversité dans la flotte, allant de la performance la plus lente ({min_val:.2f}) à la plus rapide ({max_val:.2f}).")
print("* Signification : Ceci montre l'étendue des capacités opérationnelles de la flotte. L'entreprise doit être consciente de cette dispersion pour planifier les missions (vols courts vs longs-courriers rapides).")


# ------------------------------------------------------------------------------
# 2. Consommation horaire et portée maximale
# ------------------------------------------------------------------------------
print(f"\n--- 2. Analyse de {COL_CONSO} (L/h) et {COL_PORTEE} (km) ---")

# (a) Effectuer les mêmes calculs pour la consommation horaire et la portée maximale
stats_conso = df[COL_CONSO].agg(['mean', 'median', 'std', 'min', 'max'])
stats_portee = df[COL_PORTEE].agg(['mean', 'median', 'std', 'min', 'max'])

print(f"\n(a) Statistiques descriptives ({COL_CONSO}) :")
print(stats_conso)
print(f"\n(a) Statistiques descriptives ({COL_PORTEE}) :")
print(stats_portee)

# (b) Représenter les données correspondantes à l’aide de boîtes à moustaches
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].boxplot(df[COL_CONSO], vert=True, patch_artist=True, boxprops=dict(facecolor='lightcoral'))

axes[0].set_ylabel('Consommation (L/h)')
axes[0].set_xticks([1], [COL_CONSO])

axes[1].boxplot(df[COL_PORTEE], vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))

axes[1].set_ylabel('Portée (km)')
axes[1].set_xticks([1], [COL_PORTEE])

plt.tight_layout()
plt.show() 
# 

# (c) Identifier la variable présentant la plus grande variabilité entre les avions et expliquer pourquoi cette information est importante pour l'entreprise.
cv_conso = stats_conso['std'] / stats_conso['mean']
cv_portee = stats_portee['std'] / stats_portee['mean']

print(f"\nCoefficient de Variation (CV) pour {COL_CONSO} : {cv_conso:.4f}")
print(f"Coefficient de Variation (CV) pour {COL_PORTEE} : {cv_portee:.4f}")

variable_plus_variable = COL_CONSO if cv_conso > cv_portee else COL_PORTEE
print(f"\n(c) La variable présentant la plus grande variabilité relative (CV) est : **{variable_plus_variable}**.")
print(f"Importance pour l'entreprise : Une grande variabilité dans la **{variable_plus_variable}** signifie que l'entreprise dispose d'une flotte dont les coûts opérationnels ou les capacités de mission sont très hétérogènes. Cela est crucial pour la stratégie de tarification, la planification de la maintenance et l'optimisation des itinéraires.")


# ------------------------------------------------------------------------------
# 3. Répartition des modèles d'avions
# ------------------------------------------------------------------------------
print(f"\n--- 3. Répartition des modèles d'avions ---")

# (a) Représenter la répartition des modèles d’avions à l’aide d’un diagramme en barres
repartition_modeles = df[COL_MODEL].value_counts()
print("\nRépartition des modèles :")
print(repartition_modeles)

plt.figure(figsize=(10, 6))
plt.bar(repartition_modeles.index, repartition_modeles.values, color='steelblue')

plt.xlabel(COL_MODEL)
plt.ylabel('Nombre d\'Avions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 
# 

# ------------------------------------------------------------------------------
# 4. Identification du modèle le plus représenté
# ------------------------------------------------------------------------------
print(f"\n--- 4. Identification du modèle le plus représenté ---")

# Identifier le modèle le plus représenté
modele_plus_represente = repartition_modeles.index[0]
print(f"\n(4) Le modèle le plus représenté est : **{modele_plus_represente}** ({repartition_modeles.max()} unités).")

# Discuter de l'impact potentiel de cette sur-représentation sur les résultats de l'analyse.
print("Impact potentiel : La sur-représentation de ce modèle risque de biaiser les analyses globales (moyennes, médianes, tendances) vers les caractéristiques de ce modèle. Par exemple, si ce modèle est un petit avion, les performances moyennes de l'ensemble de la flotte seront inférieures à ce qu'elles seraient dans une flotte équilibrée.")

# ==============================================================================
# FIN 5.1
# ==============================================================================
print("\n" + "="*70)
print("Section 5.1 (Analyse Univariée) Complétée.")
print("="*70)




##############5.2



# Configuration de l'affichage pour les graphiques
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# ==============================================================================
# PARTIE 0 : DÉFINITION DES VARIABLES ET CHARGEMENT (RÉPÉTITION DE LA 5.1)
# NOTE : En pratique, cette section doit être exécutée une seule fois.
# ==============================================================================
FILE_NAME = 'data-Aircraft_Performance.csv' 
CSV_SEPARATOR = ';' 
COL_VITESSE = 'Cruise_Speed_kmh' 
COL_ALTITUDE = 'Max_Altitude_ft'
COL_CONSO = 'Hourly_Fuel_Consumption_l'
COL_PORTEE = 'Max_Range_km'

try:
    df = pd.read_csv(FILE_NAME, sep=';')
    df.columns = df.columns.str.strip()
    if 'Aircraft_ID' in df.columns:
        df = df.drop(columns=['Aircraft_ID'])
    # Vérification simple (peut être retirée si la 5.1 est déjà exécutée)
    if not all(col in df.columns for col in [COL_VITESSE, COL_ALTITUDE, COL_CONSO, COL_PORTEE]):
        raise KeyError("Vérifiez les noms de colonnes dans le CSV.")
except Exception as e:
    print(f"❌ ERREUR lors du chargement des données. Veuillez vérifier le fichier et le séparateur (';' ou ','). Détail: {e}")
    exit()

# ==============================================================================
# 5.2 ANALYSE BIVARIÉE
# ==============================================================================
print("\n" + "="*70)
print("5.2 ANALYSE BIVARIÉE")
print("="*70)

# --- 1. Vitesse de croisière vs. Consommation horaire ---
print(f"\n--- 1. Étude : {COL_VITESSE} vs. {COL_CONSO} ---")

# (a) Calculer le coefficient de corrélation
corr_vitesse_conso = df[COL_VITESSE].corr(df[COL_CONSO])
print(f"\n(a) Coefficient de Corrélation (Pearson) : {corr_vitesse_conso:.4f}")
print(f"(Conclusion : Corrélation {'positive' if corr_vitesse_conso > 0 else 'négative'} et {'forte' if abs(corr_vitesse_conso) > 0.7 else 'modérée/faible'}).")

# (b) Présenter le nuage de points et tracer la droite de régression
plt.figure(figsize=(10, 6))
plt.scatter(df[COL_VITESSE], df[COL_CONSO], color='blue', alpha=0.6, label='Données')

# Calculer la droite de régression
slope, intercept, _, _, _ = stats.linregress(df[COL_VITESSE], df[COL_CONSO])
x_line = np.linspace(df[COL_VITESSE].min(), df[COL_VITESSE].max(), 100)
y_line = slope * x_line + intercept

plt.plot(x_line, y_line, color='red', label=f'Régression (r={corr_vitesse_conso:.2f})')

plt.xlabel(COL_VITESSE)
plt.ylabel(COL_CONSO)
plt.legend()
plt.grid(True)
plt.show() 
# 

# (c) Expliquer la tendance obtenue et conclure
print("\n(c) Explication : (À faire manuellement. Généralement, une tendance positive est attendue, car plus de vitesse requiert plus de puissance moteur et donc plus de consommation.)")


# --- 2. Consommation horaire vs. Portée maximale ---
print(f"\n--- 2. Étude : {COL_CONSO} vs. {COL_PORTEE} ---")

# (a) Étudier la corrélation linéaire
corr_conso_portee = df[COL_CONSO].corr(df[COL_PORTEE])
print(f"\n(a) Coefficient de Corrélation (Pearson) : {corr_conso_portee:.4f}")

# (b) Représenter cette relation par un nuage de points et tracer la droite de régression linéaire
plt.figure(figsize=(10, 6))
plt.scatter(df[COL_CONSO], df[COL_PORTEE], color='green', alpha=0.6, label='Données')

# Calculer la droite de régression
slope_cp, intercept_cp, _, _, _ = stats.linregress(df[COL_CONSO], df[COL_PORTEE])
x_line_cp = np.linspace(df[COL_CONSO].min(), df[COL_CONSO].max(), 100)
y_line_cp = slope_cp * x_line_cp + intercept_cp

plt.plot(x_line_cp, y_line_cp, color='darkorange', label=f'Régression (r={corr_conso_portee:.2f})')

plt.xlabel(COL_CONSO)
plt.ylabel(COL_PORTEE)
plt.legend()
plt.grid(True)
plt.show() 
# 

# (c) Interpréter la tendance obtenue et justifier en fonction des principes de l'aviation
print("\n(c) Interprétation et Justification : (À faire manuellement. Une corrélation positive peut indiquer que les avions à forte consommation sont les gros porteurs qui ont aussi la plus grande capacité de carburant/portée, ce qui est logique en aéronautique.)")


# --- 3. Altitude maximale vs. Vitesse de croisière ---
print(f"\n--- 3. Étude : {COL_ALTITUDE} vs. {COL_VITESSE} ---")

# (a) Étudier la relation
corr_altitude_vitesse = df[COL_ALTITUDE].corr(df[COL_VITESSE])
print(f"\n(a) Coefficient de Corrélation (Pearson) : {corr_altitude_vitesse:.4f}")

# (b) Présenter un nuage de points pour illustrer cette relation et commenter les résultats.
plt.figure(figsize=(10, 6))
plt.scatter(df[COL_ALTITUDE], df[COL_VITESSE], color='purple', alpha=0.6)

plt.xlabel(COL_ALTITUDE)
plt.ylabel(COL_VITESSE)
plt.grid(True)
plt.show() 
# 

print("\n(b) Commentaire : (À faire manuellement. Commenter la force et le sens de la corrélation.)")

# (c) Les avions volant à des altitudes plus élevées sont-ils généralement plus rapides? Expliquer pourquoi certains avions peuvent atteindre des altitudes plus élevées.
print("\n(c) Réponse et Explication : (À faire manuellement. Corrélation souvent positive, car voler haut réduit la traînée. Les avions volant haut nécessitent des moteurs plus puissants/des ailes plus fines.)")


# ==============================================================================
# FIN 5.2
# ==============================================================================
print("\n" + "="*70)
print("Section 5.2 (Analyse Bivariée) Complétée. Les interprétations sont à rédiger manuellement.")
print("="*70)


##################5.3


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Pour le graphique 3D
from sklearn.linear_model import LinearRegression # Utilisé pour le calcul du plan 3D

# ==============================================================================
# PARTIE 0 : CHARGEMENT ET PRÉPARATION DES DONNÉES (CORRECTION SYSTÉMATIQUE)
# ==============================================================================

FILE_NAME = 'data-Aircraft_Performance.csv' 
CSV_SEPARATOR = ';' 
COL_MODEL = 'Model'
COL_VITESSE = 'Cruise_Speed_kmh' 
COL_ALTITUDE = 'Max_Altitude_ft'
COL_CONSO = 'Hourly_Fuel_Consumption_l'
COL_PORTEE = 'Max_Range_km'

try:
    # 1. Chargement et nettoyage des noms de colonnes
    df = pd.read_csv(FILE_NAME, sep=CSV_SEPARATOR)
    df.columns = df.columns.str.strip()
    
    # 2. Correction des types de données (Virgule -> Point pour décimales)
    colonnes_numeriques = [COL_VITESSE, COL_ALTITUDE, COL_CONSO, COL_PORTEE]
    for col in colonnes_numeriques:
        df[col] = df[col].astype(str).str.strip().str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 3. Nettoyage des lignes et des modèles
    df.dropna(inplace=True) 
    df[COL_MODEL] = df[COL_MODEL].astype(str).str.strip()

    print("✅ Données chargées et nettoyées pour l'analyse 5.3.")

except Exception as e:
    print(f"❌ ERREUR FATALE lors du chargement des données. Détail: {e}")
    exit()

# ==============================================================================
# 5.3 ANALYSE MULTIDIMENSIONNELLE : QUESTION 1
# ==============================================================================
print("\n" + "="*70)
print("5.3 ANALYSE MULTIDIMENSIONNELLE")
print("="*70)

# --- QUESTION 1(a) : Explorer les relations entre Vitesse, Altitude et Consommation ---
print("\n--- 1(a) Exploration des corrélations triples (Vitesse, Altitude, Consommation) ---")

correlation_matrix = df[[COL_VITESSE, COL_ALTITUDE, COL_CONSO]].corr()
print("Matrice de Corrélation :")
print(correlation_matrix)

# --- QUESTION 1(b) : Présenter un nuage de points en 3 dimensions ---
print("\n--- 1(b) Nuage de Points en 3 Dimensions (Vitesse, Altitude, Consommation) ---")

try:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d') 

    # Nuage de points 3D : Vitesse(X), Altitude(Y), Consommation(Z)
    ax.scatter(df[COL_VITESSE], df[COL_ALTITUDE], df[COL_CONSO], c='blue', marker='o', alpha=0.6)

    ax.set_xlabel(COL_VITESSE + ' (X)')
    ax.set_ylabel(COL_ALTITUDE + ' (Y)')
    ax.set_zlabel(COL_CONSO + ' (Z)')
    ax.set_title('Nuage de Points 3D : Vitesse, Altitude et Consommation')

    plt.show() #
except Exception as e:
    print(f"❌ ERREUR lors de la génération du graphique 3D. Détail: {e}")

# --- QUESTION 1(c) : Interpréter les relations (via Régression Multiple) ---
print("\n--- 1(c) Interprétation des relations via Régression Multiple ---")
print("Nous utilisons la Portée comme variable dépendante pour évaluer l'influence combinée.")

# Préparation de la régression
df_reg = pd.get_dummies(df, columns=[COL_MODEL], drop_first=True)
Y = df_reg[COL_PORTEE]
colonnes_X = [COL_VITESSE, COL_CONSO, COL_ALTITUDE] + [col for col in df_reg.columns if col.startswith(f'{COL_MODEL}_')]
X = df_reg[colonnes_X].astype(float) 
X = sm.add_constant(X) 

# Entraînement du modèle OLS
model = sm.OLS(Y, X).fit()
print("\nRésumé de la Régression OLS (Portée en fonction de tout) :")
print(model.summary())

# Réponse à 1(c) : Comment ces facteurs influencent-ils la performance ? Existent-t-il des interactions significatives ?
print("\n--- RÉPONSE CHIFFRÉE À 1(c) : INFLUENCE ET SIGNIFICATION ---")
results_df = model.summary2().tables[1].copy()
results_df['Significant'] = results_df['PValue'] < 0.05
significant_factors = results_df[results_df['Significant']][['Coef', 'PValue']]
non_significant_factors = results_df[~results_df['Significant']][['Coef', 'PValue']].drop('const', errors='ignore')

if not significant_factors.empty:
    print("\nFacteurs ayant une INFLUENCE STATISTIQUEMENT SIGNIFICATIVE (P < 0.05) :")
    print(significant_factors.to_string(float_format='%.4f'))
else:
    print("\nAUCUN facteur n'est statistiquement significatif pour expliquer la Portée Maximale dans ce modèle (P > 0.05 pour tous).")
    
print(f"\nLe R-carré global est : {model.rsquared:.4f}. Ce faible score suggère que les facteurs clés (ex: capacité de carburant) sont absents du modèle.")


# ==============================================================================
# 5.3 ANALYSE MULTIDIMENSIONNELLE : QUESTION 2
# ==============================================================================

# --- QUESTION 2(a) : Analyser si certains modèles se comportent différemment ---
print("\n" + "="*70)
print("5.3 QUESTION 2 : COMPORTEMENT DES MODÈLES")
print("="*70)

# Nous utilisons les statistiques descriptives PAR MODÈLE pour comparer les comportements
comparaison_modeles = df.groupby(COL_MODEL)[[COL_VITESSE, COL_ALTITUDE, COL_CONSO]].mean()
print("\n--- 2(a) Comportement moyen des modèles (Vitesse, Altitude, Consommation) ---")
print(comparaison_modeles)

# --- QUESTION 2(b) : Identifier les modèles les plus performants globalement ---
print("\n--- 2(b) Identification des Modèles les Plus Performants Globalement ---")

# Création d'un "Score de Performance" simple : Vitesse * Portée / Consommation (Efficacité)
df['Efficacite'] = (df[COL_VITESSE] * df[COL_PORTEE]) / df[COL_CONSO]

performance_globale = df.groupby(COL_MODEL)['Efficacite'].mean().sort_values(ascending=False)

print("\nClassement d'Efficacité Globale (Vitesse * Portée / Consommation) :")
print(performance_globale.to_string(float_format='%.2f'))

print("\nInterprétation : Le modèle en tête de ce classement est le plus 'efficace'.")
print("Justification : (À rédiger manuellement) La différence d'efficacité est due à la conception (aérodynamisme, type de moteur, poids) et aux limites physiques qui rendent certains modèles plus efficaces que d'autres à différentes altitudes ou vitesses.")


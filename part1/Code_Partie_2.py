import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SAISIE DES DONNÉES (Questions 1 & 2) ---
print("## 1. Saisie des Données (XPA et YPI)")

# Données de la Production Agricole (XPA) en abscisse (x)
XPA_x = [
    100, 61, 76, 74, 90, 93, 102, 98, 103, 110, 117,
    118, 112, 115, 116, 121, 134, 130, 135  
]
# Données de la Production Industrielle (YPI) en ordonnée (y)
YPI_y = [
    10, 50, 84, 99, 113, 122, 128, 143, 145, 145, 159,
    172, 188, 204, 213, 220, 242, 254, 273  
]
n = len(XPA_x)

# Création du DataFrame avec les noms de colonnes XPA et YPI
df = pd.DataFrame({
    'XPA': XPA_x,
    'YPI': YPI_y
})

print(f"Nombre d'observations (n) = {n}")
print("Aperçu des données :")
print(df.head())
print("-" * 60)

# --- 2. STATISTIQUES DESCRIPTIVES (Questions 4, 5, 6) ---

# Question 4: Point Moyen G(XPA_barre, YPI_barre)
XPA_barre = df['XPA'].mean()
YPI_barre = df['YPI'].mean()

# Variances descriptives (pour les calculs de régression)
var_XPA = df['XPA'].var(ddof=0)
var_YPI = df['YPI'].var(ddof=0)

# Question 5: Covariance (Cov(XPA, YPI))
# ddof=0 pour la covariance descriptive (division par n)
cov_XPA_YPI = np.cov(df['XPA'], df['YPI'], ddof=0)[0, 1] 

# Question 6: Coefficient de corrélation (r)
r = df['XPA'].corr(df['YPI'])
r_carre = r**2 # Coefficient de détermination (R²)

print("## 2. Statistiques Clés (Q4, Q5, Q6)")
print(f"Point Moyen G : ({XPA_barre:.4f}, {YPI_barre:.4f})")
print(f"Variance XPA (Var(XPA)) : {var_XPA:.4f}")
print(f"Variance YPI (Var(YPI)) : {var_YPI:.4f}")
print(f"Covariance Cov(XPA, YPI) : {cov_XPA_YPI:.4f}")
print(f"Coefficient de Corrélation r : {r:.4f}")
print(f"Coefficient de Détermination R² : {r_carre:.4f}")
print("-" * 60)


# --- 3. RÉGRESSION DE YPI EN XPA (Question 8) ---
print("## 3. Régression de YPI en XPA (ŷ = a * XPA + b) - Q8")

# 8. (a) Déterminer la droite de régression
a = cov_XPA_YPI / var_XPA
b = YPI_barre - a * XPA_barre

print(f" (a) Équation de la droite (YPI en XPA) : ŷ = {a:.4f} * XPA + {b:.4f}")

# 8. (b) Valeurs ajustées (ŷ) et résidus (e)
df['YPI_ajuste'] = a * df['XPA'] + b
df['Residuel_YPI'] = df['YPI'] - df['YPI_ajuste'] 

# 8. (c) Variances (YPI en XPA)
# Var Expliquée = Var(YPI) * R²
var_expliquee_YPI = var_YPI * r_carre
# Var Résiduelle = (1/n) * Somme des carrés des résidus
somme_carres_residuelle_YPI = (df['Residuel_YPI']**2).sum()
var_residuelle_YPI = somme_carres_residuelle_YPI / n

print(f" (c) Var. Expliquée : {var_expliquee_YPI:.4f}")
print(f"     Var. Résiduelle : {var_residuelle_YPI:.4f}")
print(f"     Vérification Var(YPI) : {var_expliquee_YPI + var_residuelle_YPI:.4f} (Égale à {var_YPI:.4f})")
print("-" * 60)


# --- 4. RÉGRESSION DE XPA EN YPI (Question 9) ---
print("## 4. Régression de XPA en YPI (x̂ = a' * YPI + b') - Q9")

# 9. (a) Déterminer la droite de régression
a_prime = cov_XPA_YPI / var_YPI
b_prime = XPA_barre - a_prime * YPI_barre

print(f" (a) Équation de la droite (XPA en YPI) : x̂ = {a_prime:.4f} * YPI + {b_prime:.4f}")

# 9. (b) Valeurs ajustées (x̂) et résidus (e')
df['XPA_ajuste'] = a_prime * df['YPI'] + b_prime
df['Residuel_XPA'] = df['XPA'] - df['XPA_ajuste']

# 9. (c) Variances (XPA en YPI)
# Var Expliquée = Var(XPA) * R²
var_expliquee_XPA = var_XPA * r_carre
# Var Résiduelle = (1/n) * Somme des carrés des résidus
somme_carres_residuelle_XPA = (df['Residuel_XPA']**2).sum()
var_residuelle_XPA = somme_carres_residuelle_XPA / n

print(f" (c) Var. Expliquée : {var_expliquee_XPA:.4f}")
print(f"     Var. Résiduelle : {var_residuelle_XPA:.4f}")
print(f"     Vérification Var(XPA) : {var_expliquee_XPA + var_residuelle_XPA:.4f} (Égale à {var_XPA:.4f})")
print("-" * 60)


# --- 5. VISUALISATION (Questions 3, 8d, 9d) ---
print("## 5. Nuage de Points et Droites de Régression (Q3, Q8d, Q9d)")

plt.figure(figsize=(12, 7))

# Nuage de points (Q3)
plt.scatter(df['XPA'], df['YPI'], color='blue', marker='o', label='Observations (XPA, YPI)')

# Point Moyen G (Q4)
plt.scatter(XPA_barre, YPI_barre, color='red', marker='X', s=200, 
            label=f'Point Moyen G ({XPA_barre:.0f}, {YPI_barre:.0f})')

# Droite de régression de YPI en XPA (YPI = a*XPA + b)
XPA_range = np.linspace(df['XPA'].min() * 0.9, df['XPA'].max() * 1.1, 100)
YPI_reg_yx = a * XPA_range + b
plt.plot(XPA_range, YPI_reg_yx, color='green', linestyle='-', 
         label=f'Régression YPI/XPA: ŷ = {a:.2f}XPA + {b:.0f}')

# Droite de régression de XPA en YPI (XPA = a'*YPI + b')
# Pour l'affichage, on exprime YPI en fonction de XPA: YPI = (XPA - b') / a'
YPI_reg_xy = (XPA_range - b_prime) / a_prime
plt.plot(XPA_range, YPI_reg_xy, color='orange', linestyle='--', 
         label=f'Régression XPA/YPI: x̂ = {a_prime:.2f}YPI + {b_prime:.0f}')


plt.xlabel('Production Agricole (XPA)')
plt.ylabel('Production Industrielle (YPI)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show() 
#


##2.2

COL_ALPHA = 'alpha_deg'
COL_CL = 'Cl'
FILE_NAME = 'alpha-CL.csv'

try:
    # --- 1. Importer la partie du fichier csv correspondant. ---
    df = pd.read_csv(FILE_NAME, sep=';') # <-- Ligne corrigée
    
except FileNotFoundError:
    print(f"ERREUR : Le fichier '{FILE_NAME}' n'a pas été trouvé.")
    exit() 

# Vérification des colonnes
if COL_ALPHA not in df.columns or COL_CL not in df.columns:
    print("ERREUR : Les colonnes ne correspondent pas aux noms attendus.")
    print(f"Noms attendus : '{COL_ALPHA}' et '{COL_CL}'")
    print(f"Colonnes disponibles : {df.columns.tolist()}")
    exit()

print(f"## 1. Données '{FILE_NAME}' importées :")
print(df.head())
print("-" * 40)


# --- 2. Tracer le nuage de points (α, CL). ---
print("## 2. Affichage du Nuage de Points")

plt.figure(figsize=(8, 6))
plt.scatter(df[COL_ALPHA], df[COL_CL], color='blue', label='Données d\'essai (α, CL)')

plt.title('Nuage de Points : Coefficient de Portance (CL) vs Angle d\'Incidence (α)')
plt.xlabel('Angle d\'incidence α (alpha_deg)')
plt.ylabel('Coefficient de Portance CL')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


# --- 3. Donner les valeurs estimées de a et b (Régression CL = a * alpha + b) ---
alpha_barre = df[COL_ALPHA].mean()
CL_barre = df[COL_CL].mean()

# Calcul des coefficients de régression (méthode des moindres carrés)
# Utilisation de ddof=1 pour la covariance/variance d'échantillon (n-1)
cov_alpha_CL = df[COL_ALPHA].cov(df[COL_CL], ddof=1) 
var_alpha = df[COL_ALPHA].var(ddof=1)             

a_estime = cov_alpha_CL / var_alpha        # Pente (a)
b_estime = CL_barre - a_estime * alpha_barre # Ordonnée à l'origine (b)

print("=" * 60)
print("## 3. Coefficients de Régression Estimés")
print(f"  Coefficient de pente a : {a_estime:.4f}")
print(f"  Ordonnée à l'origine b : {b_estime:.4f}")
print(f"  Équation de la droite : CL = {a_estime:.4f} * α + {b_estime:.4f}")
print("-" * 60)


# --- 4. Calculer le coefficient de corrélation r. Conclure. ---
r = df[COL_ALPHA].corr(df[COL_CL])

print("## 4. Coefficient de Corrélation r")
print(f"  Coefficient de corrélation r : {r:.4f}")
print("  Conclusion : Le coefficient r est très proche de 1, indiquant une relation linéaire très forte et positive.")
print("-" * 60)


# --- 5. Interpréter physiquement le coefficient a. ---
print("## 5. Interprétation Physique du Coefficient a")
print(f"  Le coefficient 'a' (pente, ≈ {a_estime:.4f}) représente le **dérivé de portance** ($dC_L/d\\alpha$).")
print(f"  Il indique l'augmentation du coefficient de portance ($C_L$) pour chaque augmentation d'un degré de l'angle d'incidence ($\\alpha$).")
print("-" * 60)


# --- 6. Prédire CL pour alpha = 7°. ---
alpha_prediction = 7.0
CL_predit = a_estime * alpha_prediction + b_estime

print("## 6. Prédiction de CL pour α = 7°")
print(f"  CL prédit pour α = 7° : {CL_predit:.4f}")
print("-" * 60)


# --- Mise à jour du Graphique (Q2 et Q3 implicite) ---

# Répéter la figure pour ajouter la droite et la prédiction
plt.figure(figsize=(8, 6))
plt.scatter(df[COL_ALPHA], df[COL_CL], color='blue', label='Données d\'essai (α, CL)')

# Ajout de la droite de régression (Q3d)
alpha_range = np.linspace(df[COL_ALPHA].min(), df[COL_ALPHA].max(), 100)
CL_reg = a_estime * alpha_range + b_estime
plt.plot(alpha_range, CL_reg, color='red', linestyle='-', 
         label=f'Régression linéaire : CL = {a_estime:.2f}α + {b_estime:.2f}')

# Ajout du point de prédiction (Q6)
plt.scatter(alpha_prediction, CL_predit, color='orange', marker='s', s=100,
            label=f'Prédiction CL({alpha_prediction}°) = {CL_predit:.2f}')
# 


plt.xlabel('Angle d\'incidence α (alpha_deg)')
plt.ylabel('Coefficient de Portance CL')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


#################2.3



# --- 1. Saisie des Données du Tableau 1 ---
planetes = ['Mercure', 'Vénus', 'Terre', 'Mars', 'Jupiter', 'Saturne']
T_jours = np.array([87.97, 224.7, 365.25, 687.0, 4331.0, 10759.0]) 
a_millions_km = np.array([57.91, 108.2, 149.6, 227.9, 778.5, 1434.0])

df = pd.DataFrame({
    'Planète': planetes,
    'T (jours)': T_jours,
    'a (millions de km)': a_millions_km
})


# --- 2. Calcul des Variables Transformées (T² et a³) ---
# Y = T²
df['T² (jours²)'] = df['T (jours)']**2
# X = a³
df['a³ (millions de km)³'] = df['a (millions de km)']**3


# Coefficient de proportionnalité k = T² / a³
df['Rapport T² / a³ (k)'] = df['T² (jours²)'] / df['a³ (millions de km)³']

# Calcul du coefficient k moyen pour tracer la droite théorique
k_moyen = df['Rapport T² / a³ (k)'].mean()




# --- 3. Graphique T² en fonction de a³ (Sans étiquettes de planètes) ---
print("## 1. Graphique T² en fonction de a³")

plt.figure(figsize=(10, 6))

# Nuage de points (T² en fonction de a³)
# X est a³, Y est T²
plt.scatter(
    df['a³ (millions de km)³'],
    df['T² (jours²)'],
    color='blue',
    label='Données des planètes (T², a³)'
)

# Tracé de la droite théorique (Régression implicite passant par l'origine)
# T² = k_moyen * a³
x_max = df['a³ (millions de km)³'].max()
x_range = np.linspace(0, x_max * 1.05, 100)
y_droite_kepler = k_moyen * x_range

plt.plot(
    x_range,
    y_droite_kepler,
    color='red',
    linestyle='-',
    label=f'Loi de Kepler (T² = k * a³), k ≈ {k_moyen:.6f}'
)

# Ajout des titres et labels

plt.xlabel('Demi-grand axe au cube, a³ (millions de km)³')
plt.ylabel('Période orbitale au carré, T² (jours²)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) # Format scientifique pour les grands nombres
plt.show()
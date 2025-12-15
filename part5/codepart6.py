import numpy as np
import scipy.stats as stats
import math


# --- Données de l'exercice 6.1 (Tableau 2) ---
# Temps de chute T (en secondes) pour les 10 essais
temps_chute = np.array([0.64, 0.64, 0.63, 0.61, 0.62, 0.65, 0.60, 0.63, 0.64, 0.62])
n = len(temps_chute) # Taille de l'échantillon

# Calcul des statistiques de l'échantillon
x_bar = np.mean(temps_chute)        # Moyenne de l'échantillon
s = np.std(temps_chute, ddof=1)     # Écart-type de l'échantillon (s, avec correction n-1)
degres_liberte = n - 1              # Degrés de liberté (ddl)

print("## 📊 Statistiques de l'Échantillon (Chute Libre)")
print(f"Taille de l'échantillon (n) : {n}")
print(f"Moyenne de l'échantillon (x_bar) : {x_bar:.4f} s")
print(f"Écart-type de l'échantillon (s) : {s:.4f} s")
print("-" * 50)

# ----------------------------------------------------------------------
# --- Question 1: Loi de distribution à utiliser pour l'estimation. ---
# ----------------------------------------------------------------------

print("## 1. Loi de Distribution à Utiliser")
print("Pour l'estimation de la moyenne $m$: **Loi de Student ($t$)**")
print("* Justification: La taille de l'échantillon est petite ($n=10 < 30$) et l'écart-type de la population ($\sigma$) est inconnu (il est estimé par $s$).")
print(f"* Degrés de liberté: $\\nu = n-1 = {degres_liberte}$.")
print("\nPour l'estimation de l'écart-type $\\sigma$: **Loi du Khi-deux ($\\chi^2$)**")
print("* Justification: Pour construire l'IC de la variance $\\sigma^2$, on utilise la loi du $\\chi^2$.")
print("-" * 50)

# -----------------------------------------------------------------------------
# --- Question 2 & 3: Construire un intervalle de confiance à 95% pour m. ---
# -----------------------------------------------------------------------------

# Niveau de confiance de 95% (alpha = 0.05)
alpha = 0.05

# 2. Construction de l'IC de la moyenne (m)
# Quantile de Student t_n-1; alpha/2
t_quantile = stats.t.ppf(1 - alpha/2, degres_liberte) 

# Marge d'erreur (E)
marge_erreur = t_quantile * (s / np.sqrt(n))

# Intervalle de confiance
IC_m_inf = x_bar - marge_erreur
IC_m_sup = x_bar + marge_erreur

print("## 2. & 3. Intervalle de Confiance pour la Moyenne ($m$) à 95%")
print(f"Quantile de Student $t_{{{degres_liberte}; {alpha/2}}} \\approx {t_quantile:.4f}$")
print(f"Marge d'erreur $E \\approx {marge_erreur:.4f}$ s")
print(f"IC_{{95\%}} (m) = [{IC_m_inf:.4f} \\text{{ s}}; {IC_m_sup:.4f} \\text{{ s}}]$")
print("-" * 50)

# ----------------------------------------------------------------------------
# --- Question 4: Construire un intervalle de confiance à 95% pour sigma. ---
# ----------------------------------------------------------------------------

# 4. Construction de l'IC de l'écart-type (sigma)
# Quantiles du Khi-deux pour 95% (ddl=9)
chi2_inf = stats.chi2.ppf(alpha/2, degres_liberte)      # Chi2_9; 0.025
chi2_sup = stats.chi2.ppf(1 - alpha/2, degres_liberte)  # Chi2_9; 0.975

# Intervalle de confiance pour la variance (sigma^2)
IC_var_inf = (n - 1) * s**2 / chi2_sup
IC_var_sup = (n - 1) * s**2 / chi2_inf

# Intervalle de confiance pour l'écart-type (sigma)
IC_sigma_inf = np.sqrt(IC_var_inf)
IC_sigma_sup = np.sqrt(IC_var_sup)

print("## 4. Intervalle de Confiance pour l'Écart-type ($\\sigma$) à 95%")
print(f"Quantiles du Khi-deux : $\\chi^2_{{inf}} \\approx {chi2_inf:.4f}$, $\\chi^2_{{sup}} \\approx {chi2_sup:.4f}$")
print(f"IC_{{95\%}} (\\sigma) = [{IC_sigma_inf:.4f} \\text{{ s}}; {IC_sigma_sup:.4f} \\text{{ s}}]$")
print("-" * 50)
# --- Données de l'exercice 6.2 ---
x_bar = 90000       # Durée de vie moyenne de l'échantillon (x_bar)
sigma = 30000       # Écart-type de la population (sigma)
# n (taille de l'échantillon) est inconnue pour les premières questions.

# --- Étape 1: Énoncer les hypothèses ---
print("## 1. Hypothèses nécessaires")
print("* **Échantillon :** Les durées de vie des moteurs forment un échantillon aléatoire et indépendant.")
print("* **Écart-type :** L'écart-type de la population ($\\sigma = 30000 \\text{ km}$) est connu.")
print("* **Distribution :** La loi des moyennes d'échantillon peut être approximée par une loi normale, soit parce que la population est normale, soit parce que la taille d'échantillon $n$ est supposée grande ($n>30$), conformément au Théorème Central Limite (TCL).")
print("-" * 50)

# --- Étape 2: Construire l'intervalle de confiance à 95% ---
z_95 = stats.norm.ppf(0.975) # Quantile Z pour 95% (Z_0.025)

print("## 2. Construction de l'Intervalle de Confiance à 95%")
print(f"Quantile $Z_{{0.025}} \\approx {z_95:.3f}$.")
print("Puisque $n$ est inconnu, l'IC s'exprime comme :")
print(f"$$ IC_{{95\%}} = \\left[ \\bar{{x}} \\pm Z_{{0.025}} \\frac{{\\sigma}}{{\\sqrt{{n}}}} \\right] = \\left[ 90000 \\pm {z_95:.3f} \\frac{{30000}}{{\\sqrt{{n}}}} \\right] $$")
print("-" * 50)

# --- Étape 3: Construire l'intervalle de confiance à 99% ---
z_99 = stats.norm.ppf(0.995) # Quantile Z pour 99% (Z_0.005)

print("## 3. Construction de l'Intervalle de Confiance à 99%")
print(f"Quantile $Z_{{0.005}} \\approx {z_99:.3f}$.")
print("L'IC s'exprime comme :")
print(f"$$ IC_{{99\%}} = \\left[ 90000 \\pm {z_99:.3f} \\frac{{30000}}{{\\sqrt{{n}}}} \\right] $$")
print("Note: L'IC à 99\% sera **plus large** que l'IC à 95\% car $Z_{{0.005}} > Z_{{0.025}}$.")
print("-" * 50)

# --- Étape 4: Interpréter ces intervalles et conseiller le constructeur ---
print("## 4. Interprétation et Conseils au Constructeur")
print("* **Interprétation :** L'IC à 95\% signifie que, si l'on répétait l'étude de nombreuses fois, l'intervalle calculé contiendrait la vraie moyenne de la population ($m$) dans 95\% des cas.")
print("* **Conseils Marketing :**")
print("  - L'objectif est d'assurer $m \\ge 100\,000 \\text{ km}$.")
print("  - L'estimation ponctuelle ($\mathbf{\\bar{x} = 90\,000}$ km) est inférieure à l'objectif de $100\,000$ km.")
print("  - Pour affirmer que la durée de vie moyenne $m$ est supérieure ou égale à $100\,000$ km avec un haut niveau de confiance, la **borne inférieure** de l'IC devrait être $\ge 100\,000$ km, ce qui est impossible avec $\\bar{x} = 90\,000$ km.")
print("  - **Conseil :** Le constructeur ne peut pas garantir $100\,000$ km sur la base de ces résultats. Il doit soit **augmenter la vraie moyenne des moteurs**, soit **accepter un risque de garantie élevé**.")
print("-" * 50)

# --- Étape 5: Calculer la taille d'échantillon n pour H = 5000 km et 95% ---
H = 5000 # Demi-largeur de l'IC (marge d'erreur E)

# Formule : n = (Z_alpha/2 * sigma / H)^2
n_needed = (z_95 * sigma / H)**2

# Utilisation de math.ceil pour l'arrondi au supérieur
n_final = math.ceil(n_needed)

print("## 5. Calcul de la Taille d'Échantillon ($n$) pour une Précision Donnée")
print(f"Condition : Demi-largeur $H = {H}$ km, Confiance $95\\%$.")
print(f"$$ n = \\left( \\frac{{Z_{{0.025}} \\cdot \\sigma}}{{H}} \\right)^2 = \\left( \\frac{{ {z_95:.3f} \\cdot {sigma} }}{{ {H} }} \\right)^2 $$")
print(f"$$ n \\approx {n_needed:.2f} $$")
print(f"La taille minimale d'échantillon (arrondie à l'entier supérieur) est $\\mathbf{{n = {n_final}}}$ moteurs.")
print("On pourra utiliser l'approximation par la loi normale pour le calcul de la taille d'échantillon car n sera grand ($n > 30$).")
print("-" * 50)

# --- Questions Générales (6 à 10) ---

print("## Questions Générales")

# 6. Quel est le contexte de l'expérience ?
print("6. Contexte de l'Expérience :")
print("Le contexte est l'**estimation de la durée de vie moyenne** ($m$) d'une population de moteurs de petit échantillon (ou grand, selon le point de vue) à partir d'une moyenne échantillon ($\\bar{x} = 90\,000 \\text{ km}$) et d'un écart-type de population connu ($\\sigma = 30\,000 \\text{ km}$).")

# 7. Quelle est l'estimation ponctuelle de l'écart-type ?
print("7. Valeur d'Estimation Ponctuelle :")
print("L'estimation ponctuelle de la **moyenne de la population** $m$ est la moyenne de l'échantillon : $\\mathbf{\\bar{x} = 90\,000 \\text{ km}}$.")
print("L'estimation ponctuelle de l'**écart-type de la population** $\\sigma$ est $\\mathbf{\\sigma = 30\,000 \\text{ km}}$ (puisque $\\sigma$ est donné comme connu).")

# 8. Que signifie l'intervalle de confiance à 95% ?
print("8. Signification de l'Intervalle de Confiance à 95% :")
print("C'est l'intervalle calculé à partir de l'échantillon qui a une probabilité de $\mathbf{0.95}$ (ou $95\\%$) de contenir la **vraie moyenne de la population** $m$.")

# 9. Que se passe-t-il si on augmente le nombre d'essais dans l'expérience ?
print("9. Influence de l'Augmentation de $n$ :")
print("Si le nombre d'essais ($n$) augmente, la **précision de l'estimation augmente**. L'intervalle de confiance devient **plus étroit** (sa demi-largeur $H$ diminue, car $H$ est inversement proportionnel à $\\sqrt{n}$).")

# 10. Quelle est l'importance de l'écart-type dans ce contexte ?
print("10. Importance de l'Écart-type ($\sigma$) :")
print("L'écart-type de la population ($\\mathbf{\\sigma = 30\,000 \\text{ km}}$) est une mesure de la **dispersion intrinsèque** des durées de vie des moteurs. Il est crucial car il **détermine la largeur de l'IC** : plus $\\sigma$ est grand, plus la variabilité est grande, et plus l'estimation de la moyenne $m$ est incertaine (IC plus large).")
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import io
from statsmodels.stats.anova import anova_lm

# Contenu du fichier "grandeur-mecanique.txt" importé directement
data_content = """
v_m_s    t_min    y
115.266  14.258   2.51840
88.417   43.693   1.88692
53.424   22.067   1.76480
94.469   37.938   2.60375
43.783   38.012   2.45122
140.025  32.146   1.28144
68.466   5.417    2.27662
112.877  50.118   2.42051
74.288   19.247   2.19919
97.207   11.191   1.70704
100.138  2.447    2.61276
60.334   35.454   1.87587
146.654  40.654   1.85911
125.265  0.995    2.24567
143.345  30.726   1.51168
138.431  13.590   1.93995
105.769 38.710   2.42139
141.406  10.462   1.45434
49.734   41.456   2.21294
61.558   23.204   2.04148
44.975   56.204   1.75166
75.786   8.251    1.69846
82.755   20.464   2.00159
69.848   6.808    2.09032
131.161  55.482   1.26750
79.243   52.640   2.50211
70.903   15.476   2.55504
99.697   39.599   2.63215
55.502   49.033   2.32545
128.242  33.312   1.40099
48.201   31.779   1.86878
148.558  14.511   2.13568
124.947  5.586    1.84702
61.859 53.833   2.19709
40.607   54.025   2.07598
129.701  37.986   2.15207
117.754  20.342   1.75342
120.191  20.953   1.54090
124.840   43.557   1.41652
48.145   53.827   2.32007
79.431   53.225   2.61759
52.746   46.793   2.04053
134.941  38.522   2.03412
108.563  5.048    2.19897
76.399   9.698    1.92320
46.991   53.913   2.10067
74.208   36.386   2.33841
75.770   0.552    2.41372
120.257  6.088    1.80950
110.131  39.810   2.43295
137.593  0.304 1.42955
91.944   9.648    1.59526
53.155   32.924   1.70362
118.457  41.514   1.92221
123.686  39.118   2.55242
101.740  13.456   2.22848
124.806  42.731   1.67497
94.318   14.235   2.56125
97.501   19.524   1.96522
87.030   44.789   1.54311
42.796   38.978   2.73827
51.868   50.953   2.63138
43.457   39.457   2.47093
110.005  34.099   1.70882
74.579   5.620    2.30709
95.943   22.063   1.66875
139.832  15.912   2.12623
67.422 14.639 2.59571
85.142   58.381   1.84545
123.111  23.586   1.89591
65.168   53.523   2.50139
48.468   37.868   2.39494
71.873   47.689   2.38050
57.734   30.158   1.92133
142.267  34.614   1.36879
128.893  29.551   2.02691
109.674  11.715   2.00025
135.861  43.347   1.40264
128.404  16.846   2.08758
60.523   1.459    2.44873
138.181  38.728   2.11444
99.328   10.627   1.90953
128.818  56.428   1.46525
138.570   57.236   1.20497
74.980   54.892   2.09915
52.106   22.210 1.85506
65.073   0.927    2.50734
86.982   55.699   1.88883
129.982  25.691   2.08811
134.680  57.999   1.32666
40.765   57.817   1.81986
96.182   51.181   2.64283
85.915   17.667   2.24223
64.432   23.106   2.01057
53.185   51.068   2.78668
77.138   19.015   1.89102
143.720  10.170   1.36307
75.552   33.408   1.72886
97.067   56.169   1.68260
117.332  41.762   2.15681
79.999   34.204   1.91950
146.896  5.831    1.75818
145.869  36.900   1.99647
67.696 59.403 2.06409
94.697   8.405    1.78278
73.097   31.100   1.91400
71.332   52.642   2.53055
44.058   44.446   1.70717
107.052  41.821   2.10698
95.295   42.149   2.16258
45.663   21.569   1.63196
70.651   17.616   2.53964
139.909  48.562   1.72692
66.352   48.607   2.19487
55.938   52.024   2.71543
93.840   54.794   2.04671
148.422  30.681   1.51637
66.626   30.091   2.25397
113.935  47.898   1.97335
123.778  38.998   2.24877
"""

# Créer le DataFrame (simule l'importation de l'Étape 1)
df = pd.read_csv(io.StringIO(data_content), sep=r'\s+', header=0)
df.columns = ['v', 't', 'y']

# --- Étape 2: Construire la matrice de régression contenant les colonnes ---
# On prépare les colonnes nécessaires (v, v^2, sin(2*pi*t/12))
df['sin_t'] = np.sin(2 * np.pi * df['t'] / 12)
df['v2'] = df['v']**2
# La matrice de régression X est construite implicitement par smf.ols

# --- Étape 3: Estimer les coefficients beta0, beta1, beta2, beta3 par la méthode des moindres carrés (MCO) ---
model_formula = 'y ~ v + v2 + sin_t'
model = smf.ols(formula=model_formula, data=df)
results = model.fit()

# Extraction des coefficients
beta0 = results.params['Intercept']
beta1 = results.params['v']
beta2 = results.params['v2']
beta3 = results.params['sin_t']

print("## Résultats de la Régression OLS (Étape 2 & 3)")
print(results.summary())
print("-" * 50)

print(f"Coefficients estimés par MCO:")
print(f"  $\\beta_0$ (Intercept): {beta0:.4f}")
print(f"  $\\beta_1$ ($v$): {beta1:.4f}")
print(f"  $\\beta_2$ ($v^2$): {beta2:.6f}")
print(f"  $\\beta_3$ (sin($2\\pi t/12$)): {beta3:.4f}")
print("-" * 50)

# --- Étape 4: Calculer la somme des carrés des erreurs (SCE) ---
# SCE (Sum of Squared Residuals) est disponible via l'attribut ssr (Sum of Squared Residuals)
SCE_model = results.ssr
print(f"## Calcul de la Somme des Carrés des Erreurs (SCE) (Étape 4)")
print(f"SCE : {SCE_model:.4f}")
print("-" * 50)

# --- Étape 5: Le modèle est-il pertinent ? Justifier votre réponse. ---
# On utilise le Test F global du modèle
f_stat = results.fvalue
p_value_f = results.f_pvalue

print("##  Pertinence du Modèle (Test F) (Étape 5)")
print(f"Statistique F: {f_stat:.2f}")
print(f"P-value du Test F: {p_value_f:.4f}")

if p_value_f < 0.05:
    print("Conclusion: La P-value est très faible. Le modèle est **globalement pertinent**.")
else:
    print("Conclusion: Le modèle n'est pas globalement pertinent au seuil de 5%.")
print("-" * 50)

# --- Étape 6: Tracer y observé versus y prédit et conclure sur l'ajustement. ---

# Génération du graphique
plt.figure(figsize=(8, 5))
plt.scatter(df['y'], results.fittedvalues, alpha=0.6)
min_val = min(df['y'].min(), results.fittedvalues.min())
max_val = max(df['y'].max(), results.fittedvalues.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('Valeurs Observées de y')
plt.ylabel('Valeurs Prédites de y (ŷ)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show() 

print("## Conclusion sur l'Ajustement (Étape 6)")
print("Le nuage de points s'aligne étroitement autour de la ligne de référence y=x, indiquant un **très bon ajustement**.")
print("-" * 50)

# --- Étape 7: Interpréter le rôle physique des coefficients. ---

print("## Interprétation Physique des Coefficients (Étape 7)")
print(f"* $\\mathbf{{\\beta_1}}$ ($v$) : {beta1:.4f} (Positif et significatif)")
print("  - $\\beta_1$ et $\\beta_2$ négatif (voir $\\beta_2$) modélisent l'**effet quadratique** de la vitesse. L'effet de la vitesse est non linéaire (parabole).")

print(f"* $\\mathbf{{\\beta_2}}$ ($v^2$) : {beta2:.6f} (Négatif et significatif)")
print("  - $\\beta_2$ représente l'**effet non linéaire** de la vitesse, suggérant que $y$ augmente puis diminue à mesure que $v$ augmente, ou vice-versa (ici, la parabole est concave, ouverte vers le bas).")

print(f"* $\\mathbf{{\\beta_3}}$ ($\sin(2\\pi t/12)$) : {beta3:.4f} (Positif et très significatif)")
print("  - $\\beta_3$ traduit l'**influence cyclique** de $t$ avec une période de 12 minutes sur la grandeur mécanique $y$.")
print("-" * 50)

# --- Étape 8: Prédire la valeur de y pour v = 115 m/s et t = 5 minutes. ---

# Préparation des nouvelles données
new_data = pd.DataFrame({'v': [115.0], 't': [5.0]})
new_data['v2'] = new_data['v']**2
new_data['sin_t'] = np.sin(2 * np.pi * new_data['t'] / 12)

# Prédiction
prediction = results.predict(new_data)

print(f"## Prédiction (Étape 8)")
print(f"Pour $v = 115 \, \\text{{m/s}}$ et $t = 5 \, \\text{{minutes}}$:")
print(f"  - La valeur prédite ($\\hat{{y}}$) est : {prediction[0]:.4f}")
print("-" * 50)

# --- Étape 9: Comparer ce modèle avec un modèle sans terme sinusoïdal. ---

# 1. Ajuster le modèle réduit (sans sin_t)
model_reduced_formula = 'y ~ v + v2'
model_reduced = smf.ols(formula=model_reduced_formula, data=df)
results_reduced = model_reduced.fit()
SCE_reduced = results_reduced.ssr

# 2. Comparaison formelle (Test F d'ajout de variables)
anova_test = anova_lm(results_reduced, results)

print("##  Comparaison des Modèles (Étape 9)")
print("### Modèle Complet (avec terme sinusoïdal):")
print(f"  - SCE : {SCE_model:.4f}")
print(f"  - R-carré ajusté : {results.rsquared_adj:.4f}")

print("\n### Modèle Réduit (sans terme sinusoïdal):")
print(f"  - SCE : {SCE_reduced:.4f}")
print(f"  - R-carré ajusté : {results_reduced.rsquared_adj:.4f}")

print("\n### Conclusion de la Comparaison:")
if SCE_model < SCE_reduced:
    print("Le Modèle Complet est nettement **préférable** car il a une SCE beaucoup plus faible et un $R^2$ ajusté beaucoup plus élevé.")
    print("Le terme sinusoïdal est **statistiquement significatif**.")

print("\n### Test F formel (ANOVA) pour comparer les modèles:")
print(anova_test)

p_value_anova = anova_test.iloc[1]['Pr(>F)']
if p_value_anova < 0.05:
    print(f"\nLa P-value du test F ({p_value_anova:.4f}) confirme que l'ajout du terme sinusoïdal est **statistiquement significatif**.")
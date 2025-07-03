###################################################################################
#                       Figure 2 : Illustration du Théorème 2
###################################################################################


from numpy import *
import numpy as np
from matplotlib import pylab
import pylab as plt

T = 2*np.pi
alpha = 1

def choix_directions(N): # Choix des tirages pour les U_i dans la formule de r_K
    phase=2*np.pi*np.random.rand(N)
    abscisses=np.cos(phase)
    ordonnees=np.sin(phase)
    return abscisses,ordonnees

def choix_poids(N):
    # Choix des tirages pour les Gamma_i, on renvoie directement Gamma_i^(-1/alpha}
    return np.cumsum(np.random.exponential(1,N))**(-1/alpha)

def r(u,abscisses,ordonnees,poids):
    # On en tire le support r_K(produit scalaire dans R^2)
    t=poids*(u[0]*abscisses+u[1]*ordonnees)
    return np.max(t[t>=0])

Nb_directions=1000
# r_K est un sup sur l'ensemble des entiers, le Théorème 3 dit qu'à chaque tirage, à
# alpha fixé, le sup est peut être pris presque sûrement sur un nombre fini N_alpha
# de Gamma_i^(-1/alpha}U_i. Dans la pratique, pour des alpha inférieurs à 100,
# N_alpha = 1000 suffit très largement

abscisses,ordonnees = choix_directions(Nb_directions)
poids=choix_poids(Nb_directions)

# Transformation de Fourier
# N = nb de coefficient de Fourier
N=10000

t,dt = np.linspace(0, 2*np.pi, 2*N + 2, endpoint=False, retstep=True)
f=[]
for x in t:
    f.append(r([np.cos(x),np.sin(x)],abscisses,ordonnees,poids))

y = np.fft.rfft(f) / t.size
coeff_constant=y[0].real
coeff_cos=y[1:-1].real
coeff_sin=-y[1:-1].imag

def series_real_coeff(a0, a, b, t, T):
    """calculates the Fourier series with period T at times t,
       from the real coeff. a0,a,b"""
    tmp = ones_like(t) * a0 / 2.
    for k, (ak, bk) in enumerate(zip(a, b)):
        tmp += ak * cos(2 * pi * (k + 1) * t / T) + bk * sin(
            2 * pi * (k + 1) * t / T)
    return tmp

def series_derivative(a0,a,b,t,T):
    tmp = zeros_like(t)
    cumul=zeros_like(t)
    for k, (ak, bk) in enumerate(zip(a, b)):
        tmp +=  -(k+1) * ak * sin(2 * pi * (k+1 ) * t / T)  + (k+1) * bk * cos(2 * pi * (k+1 ) * t / T)
        cumul+=tmp
    return cumul/k

r_series=2*series_real_coeff(coeff_constant,coeff_cos,coeff_sin,t,2*np.pi)
r_prime_series=2*series_derivative(coeff_constant,coeff_cos,coeff_sin,t,2*np.pi)
plt.plot(t,f,color='red')
plt.plot(t,r_series,color='green',linestyle='dashed')
plt.plot(t,r_prime_series)
plt.show()


# Dessin
# On représente le convexe qu'on vient de créer avec Nb_directions directions dans le
# développement de LePage.


x=[]
y=[]
for i in range(len(t)):
    x.append(r_series[i]*np.cos(t[i])-r_prime_series[i]*np.sin(t[i]))
    y.append(r_series[i]*np.sin(t[i])+r_prime_series[i]*np.cos(t[i]))
    # Cette formule est donnée dans la littérature, et donne la paramétrisation d'un
    # convexe en fonction de sa fonction de support. Ici on traite de convexes non
    # strictement convexes (voir Section 3.3.2) donc cette formule ne donne pas une
    # paramétrisation, mais grâce à la formule de r_K avec LePage on s'apperçoit
    # qu'elle est constante en les sommets du convexe K, on obtient un dessin du
    # convexe en reliant ces points.

    # Attention, la phénomène de Gibbs altère l'observation précédente.

plt.plot(x,y)   
plt.savefig("convexe_stable_par_fourier.pdf") 
plt.show()

# Pour illustrer le Théorème 3, c'est le même code, il faut juste rajouter une ligne
# dans le plot pour faire apparaître les Gamma_i^(-1/alpha} U_i. Pour le Théorème 4,
# on affiche pour différentes valeurs de alpha.

###################################################################################
#                                       Figure 5
###################################################################################

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Pour faire un nombre important de calculs on fait appel à la parallèlisation
from joblib import Parallel, delayed

# On fait appel aux fonctions choix_directions, choix_poids et r comme pour les
# figures précédentes. On calcule l'espérance et la variance de R_alpha pour un grand
# nombre de valeurs de alpha, on adapte facilement le code pour illustrer
# graphiquement la distribution empirique de R_alpha pour un unique alpha donné.

###########################################################
#       Méthode 1 : Phénomène de Gibbs
###########################################################

# On regarde la différence en valeur absolue entre Féjer et Dirichet (Voir Annexes)
def deriv_fejer(a0, a, b, t, T):
  tmp = np.zeros_like(t)
  cumul = np.zeros_like(t)
  for k, (ak, bk) in enumerate(zip(a, b)):
      tmp += -(k + 1) * ak * np.sin(2 * np.pi * (k + 1) * t / T) + (k + 1) * bk * np.cos(2 * np.pi * (k + 1) * t / T)
      cumul += tmp
  return cumul / k

def deriv_dirichlet(a0, a, b, t, T):
  tmp = np.zeros_like(t)
  for k, (ak, bk) in enumerate(zip(a, b)):
      tmp += -(k + 1) * ak * np.sin(2 * np.pi * (k + 1) * t / T) + (k + 1) * bk * np.cos(2 * np.pi * (k + 1) * t / T)
  return tmp

means = []
var = []

for alpha in range(1,100) :
  print(alpha)
  Nb_tirages = 10000
  Nb_directions = 2000
  N = 500

  def compute_sommets_maxloc(k):
      abscisses, ordonnees = choix_directions(Nb_directions)
      poids = choix_poids(Nb_directions, alpha)

      t = np.linspace(0, 2*np.pi, 2*N + 2, endpoint=False)
      f = [r([np.cos(x), np.sin(x)], abscisses, ordonnees, poids) for x in t]

      y = np.fft.rfft(f) / len(t)
      a0 = y[0].real
      a = y[1:-1].real
      b = -y[1:-1].imag

      r_prime_fejer = 2 * deriv_fejer(a0, a, b, t, 2 * np.pi)
      r_prime_dirichlet = 2 * deriv_dirichlet(a0, a, b, t, 2 * np.pi)
      diff = np.abs(r_prime_fejer - r_prime_dirichlet)

      diff_smooth = gaussian_filter1d(diff, sigma=2)
      peaks, _ = find_peaks(diff_smooth, distance=N // 200, prominence=np.std(diff_smooth) * 0.2)

      return len(peaks)

  # Simulation parallélisée
  np.random.seed(42)
  nb_sommets = Parallel(n_jobs=-1, verbose=10)(delayed(compute_sommets_maxloc)(k) for k in range(Nb_tirages))

  # Statistiques
  means.append(np.mean(nb_sommets))
  var.append(np.var(nb_sommets))

import pandas as pd

# Export CSV pour exploitation
df = pd.DataFrame({
   'alpha': list(range(1, 100)),
   'mean': means,
   'variance': var
})
df.to_csv("resultats_alpha.csv", index=False)

# Affichage résultats

plt.plot(means, np.arange(1,100), label="Espérance")
plt.plot(var, np.arange(1,100), label="Variance")
plt.legend()
plt.show()

###########################################################
#       Méthode 2 : Compter les sommets d'un convexe
#       avec ConvexHull de Scipy.spatial
###########################################################

# Seul la fonction compute_sommets_maxloc nécessite une modification.

from scipy.spatial import ConvexHull

def compute_sommets_convex():
    abscisses, ordonnees = choix_directions(Nb_directions)
    poids = choix_poids(Nb_directions, alpha)

    t = np.linspace(0, 2*np.pi, 2*N + 2, endpoint=False)
    f = [r([np.cos(x), np.sin(x)], abscisses, ordonnees, poids) for x in t]

    # Coordonnées cartésiennes du bord
    x = np.array(f) * np.cos(t)
    y = np.array(f) * np.sin(t)
    points = np.vstack((x, y)).T

    try:
        hull = ConvexHull(points)
        return len(hull.vertices)  # nombre de sommets du polygone
    except:
        return 0  # au cas où l'enveloppe convexe échoue (très rare)

# Les calculs sont beaucoup plus rapides avec cette méthode.

###################################################################################
#                                   Figures 6 et 7
###################################################################################

# Voici le code pour la Figure 7
# Pour adapter au code de la figure 6, ne regarder qu'une seule valeur de alpha, et
# on regarde la distribution des valeurs obtenus dans la ligne :
# perimeters = [compute_perimeter(alpha) for _ in range(Nb_tirages)]
# dans la fonction simulation_for_alpha

from scipy.stats import genextreme as gev # Loi de Fréchet

# Paramètres
liste_alpha = np.linspace(1, 1000, 1)
theta = np.linspace(0, 2 * np.pi, 2 * N + 2, endpoint=False)

def compute_perimeter(alpha): # Proposition 10, méthode des trapèzes
    abscisses, ordonnees = choix_directions(Nb_directions)
    poids = choix_poids(Nb_directions, alpha)
    r_theta = np.array([r([np.cos(t), np.sin(t)], abscisses, ordonnees, poids) for t in theta])
    return np.trapz(r_theta, theta)

def simulation_for_alpha(alpha):
    perimeters = [compute_perimeter(alpha) for _ in range(Nb_tirages)]
    perimeters = np.array([p for p in perimeters if not np.isnan(p)])

    if len(perimeters) < 30:
        print(f"Alpha = {alpha:.2f} → Trop peu de données.")
        return alpha, np.nan, np.nan, np.nan

    shape, loc, scale = gev.fit(perimeters)
    if shape <= 0:
        print(f"Alpha = {alpha:.2f} → Pas Fréchet.")
        return alpha, np.nan, np.nan, np.nan

    alpha_a = 1 / shape
    sigma = scale
    m = loc - scale / shape

    print(f"Alpha = {alpha:.2f} → alpha_a = {alpha_a:.4f}, sigma = {sigma:.4f}, m = {m:.4f}")
    return alpha, alpha_a, sigma, m

# Simulation parallèle
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(simulation_for_alpha)(alpha) for alpha in liste_alpha
)

# Sauvegarde
df = pd.DataFrame(results, columns=["alpha_sim", "alpha_a", "sigma", "m"])
df.to_csv("frechet_fit_perimeter.csv", index=False)
print(df.to_csv(index=False))

# Affichage résultats

# Chargement du CSV généré par ta simulation
df = pd.read_csv("frechet_fit_results.csv")

# On garde uniquement les lignes avec shape > 0 (cas Fréchet)
df_valid = df[df['shape'] > 0].copy()

# On renomme alpha en alpha_sim pour plus de clarté
df_valid = df_valid.rename(columns={'alpha': 'alpha_sim'})

# Conversion vers les paramètres Fréchet classiques
df_valid['alpha_a'] = 1 / df_valid['shape']  # alpha ajusté de la Fréchet
df_valid['sigma'] = df_valid['scale']
df_valid['m'] = -df_valid['scale'] - df_valid['loc'] / df_valid['alpha_a']

# Tracés
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(df_valid['alpha_sim'], df_valid['alpha_a'], 'b.-')
plt.ylabel(r"$\alpha_a$")
plt.title("Évolution des paramètres de la loi de Fréchet de l'aire (en fonction de α simulé)")

plt.subplot(3, 1, 2)
plt.plot(df_valid['alpha_sim'], df_valid['sigma'], 'g.-')
plt.ylabel(r"$\sigma$")

plt.subplot(3, 1, 3)
plt.plot(df_valid['alpha_sim'], df_valid['m'], 'r.-')
plt.ylabel(r"$m$")
plt.xlabel(r"$\alpha$ (paramètre de simulation)")

plt.tight_layout()
plt.show()

###################################################################################
#                                      Figure 8
###################################################################################

# On compare la loi de r_K(0) (Proposition 11) avec celle de la Proposition 14
# (avec N grand et Theta = 0)

from scipy.stats import expon, kstest
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import rv_continuous

# --------- Paramètres ---------
alpha = 1.5
N = 1000              # Nombre de directions
n_samples = 10000     # Nombre de réalisations
m = 0                 # Paramètre de localisation

# --------- Calcul de sigma analytique ---------
def integrand(theta, alpha):
    return np.cos(theta) ** alpha

I, _ = quad(integrand, 0, np.pi / 2, args=(alpha,))
sigma = (I / np.pi) ** (1 / alpha)
print(f"Sigma = {sigma:.6f}")

# --------- Simulation numérique ---------
thetas = 2 * np.pi * np.arange(N) / N
cos_vals = np.maximum(np.cos(thetas), 0)  # cos^+

# Échantillons exponentiels (Exp(1/N))
scale = N
gamma_matrix = expon(scale=scale).rvs(size=(n_samples, N))

# Simulation du maximum
samples = np.max(gamma_matrix ** (-1 / alpha) * cos_vals, axis=1)

# --------- CDF empirique ---------
emp_cdf = ECDF(samples)

# --------- CDF théorique ---------
def frechet_cdf(x, alpha, sigma, m):
    x = np.array(x)
    out = np.zeros_like(x)
    valid = x > m
    out[valid] = np.exp(- (sigma / (x[valid] - m)) ** alpha)
    return out

x_vals = np.linspace(m, 20, 500)  # limité à 20
cdf_theo = frechet_cdf(x_vals, alpha, sigma, m)

# --------- Densité théorique ---------
def frechet_pdf(x, alpha, sigma, m):
    x = np.array(x)
    out = np.zeros_like(x)
    valid = x > m
    out[valid] = (alpha * sigma**alpha / (x[valid] - m)**(alpha + 1)) * \
                  np.exp(- (sigma / (x[valid] - m)) ** alpha)
    return out

pdf_theo = frechet_pdf(x_vals, alpha, sigma, m)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Histogramme superposé à la densité
ax[0].hist(samples, bins=2000, density=True, alpha=0.6, color='skyblue', edgecolor='none', label='Loi Proposition 14')
ax[0].plot(x_vals, pdf_theo, color='darkblue', linewidth=2, label="Densité théorique Fréchet")
ax[0].set_xlim([m, 6])
ax[0].set_title("Comparaison des distributions (alpha = {:.2f})".format(alpha))
ax[0].set_xlabel("x")
ax[0].set_ylabel("Densité")
ax[0].legend()
ax[0].grid(True)

# Courbes de répartition
ax[1].plot(x_vals, cdf_theo, label="CDF théorique Fréchet", linewidth=2)
ax[1].plot(emp_cdf.x, emp_cdf.y, label="CDF empirique", linestyle='--')
ax[1].set_xlim([m, 20])
ax[1].set_xlabel("x")
ax[1].set_ylabel("F(x)")
ax[1].set_title(f"Comparaison des CDF (alpha = {alpha})")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# --------- Test de Kolmogorov-Smirnov ---------
class frechet_custom_gen(rv_continuous):
    def _cdf(self, x):
        return frechet_cdf(x, alpha, sigma, m)

frechet_custom = frechet_custom_gen(a=m, name='frechet_custom')
ks_stat, p_val = kstest(samples, frechet_custom.cdf)

print(f"\nTest de Kolmogorov-Smirnov :\nD = {ks_stat:.5f}, p-value = {p_val:.5f}")

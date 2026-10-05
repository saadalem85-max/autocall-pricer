"""
Pricer d'un autocall Athena par Monte Carlo.

Produit (nominal 100, sous-jacent unique, observations annuelles) :
  - A chaque date d'observation i (annee i) : si S_i / S_0 >= niveau de rappel,
    le produit est rembourse par anticipation : 100 * (1 + coupon * i).
  - A maturite, s'il n'a jamais ete rappele :
        S_T / S_0 >= barriere  -> remboursement de 100 (capital protege)
        S_T / S_0 <  barriere  -> remboursement de 100 * S_T / S_0 (perte en capital)

Modele : Black-Scholes (taux, dividende et volatilite constants).
"""

import numpy as np
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Black-Scholes : formules fermees (servent de reference pour le Monte Carlo)
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, r, q, sigma, kind="call"):
    """Prix d'une option europeenne."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S, K, T, r, q, sigma, kind="call"):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if kind == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def bs_vega(S, K, T, r, q, sigma):
    """Sensibilite du prix a la volatilite (pour +1.00 de vol, soit 100 points)."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


# ---------------------------------------------------------------------------
# 2. Moteur Monte Carlo
# ---------------------------------------------------------------------------

def simulate_paths(S0, r, q, sigma, obs_times, n_paths=200_000, seed=42):
    """
    Simule le sous-jacent aux dates d'observation uniquement.

    Le mouvement brownien geometrique a une solution exacte, donc inutile de
    discretiser finement : un pas par date d'observation suffit.
    Variables antithetiques (Z et -Z) pour reduire la variance.

    Retourne un tableau (n_paths, n_obs).
    """
    rng = np.random.default_rng(seed)
    obs_times = np.asarray(obs_times, dtype=float)
    dt = np.diff(np.concatenate(([0.0], obs_times)))

    half = n_paths // 2
    Z = rng.standard_normal((half, len(obs_times)))
    Z = np.vstack([Z, -Z])

    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return S0 * np.exp(np.cumsum(log_increments, axis=1))


def mc_european(S0, K, T, r, q, sigma, kind="call", n_paths=200_000, seed=42):
    """Option europeenne par Monte Carlo : test de validation du moteur."""
    ST = simulate_paths(S0, r, q, sigma, [T], n_paths, seed)[:, 0]
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    pv = np.exp(-r * T) * payoff
    return pv.mean(), pv.std(ddof=1) / np.sqrt(len(pv))


# ---------------------------------------------------------------------------
# 3. Autocall Athena
# ---------------------------------------------------------------------------

def price_athena(coupon, S0=100.0, r=0.03, q=0.02, sigma=0.20, maturity=5,
                 autocall_level=1.0, barrier=0.6, n_paths=200_000, seed=42):
    """
    Prix de l'Athena en % du nominal.

    Retourne (prix, erreur standard, probabilites de rappel par annee,
              probabilite de perte en capital).
    """
    obs_times = np.arange(1, maturity + 1)
    perf = simulate_paths(S0, r, q, sigma, obs_times, n_paths, seed) / S0

    pv = np.zeros(len(perf))                    # valeur actualisee par trajectoire
    alive = np.ones(len(perf), dtype=bool)      # trajectoires pas encore rappelees
    call_probs = []

    for i, t in enumerate(obs_times):
        called = alive & (perf[:, i] >= autocall_level)
        pv[called] = 100.0 * (1 + coupon * t) * np.exp(-r * t)
        call_probs.append(called.mean())
        alive &= ~called

    # Trajectoires jamais rappelees : remboursement a maturite
    final = perf[:, -1]
    redemption = np.where(final >= barrier, 100.0, 100.0 * final)
    pv[alive] = redemption[alive] * np.exp(-r * maturity)

    loss_prob = (alive & (final < barrier)).mean()
    price = pv.mean()
    stderr = pv.std(ddof=1) / np.sqrt(len(pv))
    return price, stderr, np.array(call_probs), loss_prob


def par_coupon(**kwargs):
    """
    Coupon annuel qui met le produit au pair (prix = 100).

    Le prix est lineaire en coupon : prix(c) = prix(0) + c * pente.
    Avec les memes trajectoires (meme seed), deux evaluations suffisent.
    """
    p0 = price_athena(0.0, **kwargs)[0]
    p1 = price_athena(1.0, **kwargs)[0]
    return (100.0 - p0) / (p1 - p0)


# ---------------------------------------------------------------------------
# 4. Analyse
# ---------------------------------------------------------------------------

def main():
    S0, r, q, sigma, T = 100.0, 0.03, 0.02, 0.20, 5

    print("=== 1. Validation du moteur Monte Carlo (call ATM 1 an) ===")
    bs = bs_price(S0, 100, 1, r, q, sigma)
    mc, se = mc_european(S0, 100, 1, r, q, sigma)
    print(f"Black-Scholes : {bs:.4f}")
    print(f"Monte Carlo   : {mc:.4f} +/- {1.96 * se:.4f} (IC 95 %)")
    print(f"Delta : {bs_delta(S0, 100, 1, r, q, sigma):.4f}   "
          f"Vega : {bs_vega(S0, 100, 1, r, q, sigma):.4f}")

    print("\n=== 2. Athena 5 ans, rappel 100 %, barriere 60 % ===")
    c = par_coupon(S0=S0, r=r, q=q, sigma=sigma, maturity=T)
    price, se, probs, loss = price_athena(c, S0=S0, r=r, q=q, sigma=sigma, maturity=T)
    print(f"Coupon au pair : {100 * c:.2f} % par an")
    print(f"Prix a ce coupon : {price:.2f} +/- {1.96 * se:.2f}")
    for year, p in enumerate(probs, start=1):
        print(f"  Probabilite de rappel annee {year} : {100 * p:.1f} %")
    print(f"  Probabilite de perte en capital : {100 * loss:.1f} %")

    print("\n=== 3. Sensibilites (coupon fige au niveau du pair) ===")
    vols = np.linspace(0.10, 0.40, 13)
    prices_vol = [price_athena(c, S0=S0, r=r, q=q, sigma=v, maturity=T)[0] for v in vols]
    barriers = np.linspace(0.40, 0.80, 9)
    prices_bar = [price_athena(c, S0=S0, r=r, q=q, sigma=sigma, maturity=T, barrier=b)[0]
                  for b in barriers]
    coupons_vol = [par_coupon(S0=S0, r=r, q=q, sigma=v, maturity=T) for v in vols]

    # Vega par difference finie : variation du prix pour +1 point de vol
    up = price_athena(c, S0=S0, r=r, q=q, sigma=sigma + 0.01, maturity=T)[0]
    down = price_athena(c, S0=S0, r=r, q=q, sigma=sigma - 0.01, maturity=T)[0]
    print(f"Vega (pour +1 point de vol) : {(up - down) / 2:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(100 * vols, prices_vol, marker="o")
    axes[0].set(title="Prix vs volatilite", xlabel="Volatilite (%)", ylabel="Prix (% nominal)")
    axes[1].plot(100 * barriers, prices_bar, marker="o", color="tab:red")
    axes[1].set(title="Prix vs barriere", xlabel="Barriere (% du niveau initial)",
                ylabel="Prix (% nominal)")
    axes[2].plot(100 * vols, 100 * np.array(coupons_vol), marker="o", color="tab:green")
    axes[2].set(title="Coupon au pair vs volatilite", xlabel="Volatilite (%)",
                ylabel="Coupon annuel (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("sensibilites.png", dpi=150)
    print("Graphiques enregistres dans sensibilites.png")


if __name__ == "__main__":
    main()

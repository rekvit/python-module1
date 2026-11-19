import math
from statistics import NormalDist
import pandas as pd

z = NormalDist().inv_cdf


def n_per_group(
    sigma,
    mde,
    alpha=0.05,
    power=0.8,
    comparisons=1,
):
    alpha_adj = alpha / comparisons

    z_alpha = z(1 - alpha_adj / 2)
    z_power = z(power)

    n = (2 * sigma**2 * (z_alpha + z_power) ** 2 / mde**2)

    return math.ceil(n)


def mde_for_n(
    sigma,
    n,
    alpha=0.05,
    power=0.8,
    comparisons=1,
):
    alpha_adj = alpha / comparisons

    z_alpha = z(1 - alpha_adj / 2)
    z_power = z(power)

    mde = (math.sqrt(2) * sigma * (z_alpha + z_power) / math.sqrt(n))

    return mde

    
def diff_ci(t: pd.Series, c: pd.Series, alpha: float = 0.05) -> dict:
    """Разность средних test − control, её 95 % интервал и p-значение."""
    d = t.mean() - c.mean()
    # Стандартная ошибка разности: дисперсия каждой группы делится на её размер.
    se = (t.var() / len(t) + len(c)) ** 0.5
    q = NormalDist().inv_cdf(1 - alpha / 2)
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))
    return {"diff": d, "lo": d - q * se, "hi": d + q * se, "p": p}


def srm_pvalue(sizes) -> float:
    # p-значение chi2 согласия с равными долями для 2 или 3 групп
    n = sum(sizes)
    exp = n / len(sizes)
    chi2 = sum((s - exp) ** 2 / exp for s in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    if len(sizes) == 3:
        return math.exp(-chi2 / 2)
    raise ValueError("для 4 и более групп — scipy.stats.chisquare")


def cuped(y, x):
    # метрика после поправки CUPED и theta; y и x — pandas.Series одной длины
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta



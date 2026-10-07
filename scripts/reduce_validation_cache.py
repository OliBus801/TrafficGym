"""Reduce a full validation_cache_*.npz to the few arrays/scalars the main-paper
fidelity strip needs, so the figure can be rebuilt locally without moving
hundreds of megabytes of per-link and per-trip data."""
import sys

import numpy as np
from scipy.stats import wasserstein_distance

src, dst = sys.argv[1], sys.argv[2]
c = np.load(src)
n_seeds = len(c["seeds"].tolist())

m_en, t_en = c["matsim_en_route"], c["torchdnl_en_route"]
m_lv, t_lv = c["matsim_link_vol"], c["torchdnl_link_vol"]
m_tt, t_tt = c["matsim_trip_tt"], c["torchdnl_trip_tt"]


def geh(model, count):
    model = np.asarray(model, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    denom = model + count
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sqrt(2.0 * (model - count) ** 2 / denom)
    return np.where(denom == 0, 0.0, v)


def geh_series(model_by_seed, ref_by_seed):
    ref = np.asarray(ref_by_seed, dtype=np.float64).mean(axis=0)
    return geh(np.asarray(model_by_seed, dtype=np.float64), ref[None, :]).ravel()


g_en = geh_series(t_en, m_en)
g_lv = geh_series(t_lv, m_lv)
mv, tv = m_lv.mean(axis=0), t_lv.mean(axis=0)
used = (mv > 0) | (tv > 0)
r = float(np.corrcoef(mv, tv)[0, 1])
w = float(wasserstein_distance(np.asarray(t_tt, dtype=np.float64).ravel(),
                               np.asarray(m_tt, dtype=np.float64).ravel()))

np.savez_compressed(
    dst,
    bins=c["bins"],
    matsim_en_route_med=np.median(m_en, axis=0),
    torchdnl_en_route_med=np.median(t_en, axis=0),
    n_links=np.int64(m_lv.shape[1]),
    n_seeds=np.int64(n_seeds),
    n_loaded_links=np.int64(used.sum()),
    trips_per_seed=np.int64(len(t_tt) // n_seeds),
    en_route_geh_max=np.float64(g_en.max()),
    en_route_geh_mean=np.float64(g_en.mean()),
    en_route_geh_pct_lt5=np.float64(np.mean(g_en < 5.0) * 100.0),
    link_geh_max=np.float64(g_lv.max()),
    link_geh_pct_lt5=np.float64(np.mean(g_lv < 5.0) * 100.0),
    link_obs=np.int64(g_lv.size),
    pearson_r=np.float64(r),
    wasserstein_s=np.float64(w),
    matsim_mean_tt=np.float64(np.asarray(m_tt, dtype=np.float64).mean()),
    torchdnl_mean_tt=np.float64(np.asarray(t_tt, dtype=np.float64).mean()),
)
print(f"{src} -> {dst}: r={r:.6f} W={w:.2f}s maxGEH_en={g_en.max():.2f} "
      f"maxGEH_lv={g_lv.max():.2f} loaded={used.sum():,} trips/seed={len(t_tt)//n_seeds:,}")

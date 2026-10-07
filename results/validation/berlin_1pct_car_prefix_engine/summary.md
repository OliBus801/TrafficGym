# TorchDNL vs MATSim — DNL-fidelity validation summary

Seeds compared: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

## GEH statistic (replaces MSE/MAPE)

| series | n | mean | median | max | % GEH<5 | % GEH<10 |
|---|---|---|---|---|---|---|
| en_route | 2880 | 30.49 | 10.89 | 123.98 | 45.8% | 49.5% |
| departures | 2880 | 3.39 | 0.24 | 137.41 | 84.6% | 97.6% |
| arrivals | 2880 | 4.08 | 0.53 | 136.82 | 77.3% | 97.5% |
| link_volume | 3180780 | 0.68 | 0.09 | 88.41 | 97.5% | 98.7% |

## Envelope / self-consistency (en-route series)

- MATSim leave-one-out self-consistency (fraction of a held-out MATSim seed inside the envelope of the other MATSim seeds): **75.0%**
- TorchDNL fraction inside MATSim's inter-seed envelope: **14.2%**
- MATSim envelope width on loaded bins (median, relative to the median series): **0.2%**. An inclusion rate is only meaningful against this: the narrower the envelope, the harsher both numbers above, so they must be compared across scenarios only together.

## Pooled trip travel-time distribution

- Wasserstein distance: **403.46s**
- KS statistic: 0.0162 (p=0)

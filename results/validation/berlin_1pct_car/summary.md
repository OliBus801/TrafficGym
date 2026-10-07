# TorchDNL vs MATSim — DNL-fidelity validation summary

Seeds compared: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

## GEH statistic (replaces MSE/MAPE)

| series | n | mean | median | max | % GEH<5 | % GEH<10 |
|---|---|---|---|---|---|---|
| en_route | 2880 | 0.08 | 0.03 | 0.74 | 100.0% | 100.0% |
| departures | 2880 | 0.06 | 0.02 | 0.65 | 100.0% | 100.0% |
| arrivals | 2880 | 0.09 | 0.03 | 0.83 | 100.0% | 100.0% |
| link_volume | 3180780 | 0.01 | 0.00 | 1.41 | 100.0% | 100.0% |

## Envelope / self-consistency (en-route series)

- MATSim leave-one-out self-consistency (fraction of a held-out MATSim seed inside the envelope of the other MATSim seeds): **75.0%**
- TorchDNL fraction inside MATSim's inter-seed envelope: **61.0%**
- MATSim envelope width on loaded bins (median, relative to the median series): **0.2%**. An inclusion rate is only meaningful against this: the narrower the envelope, the harsher both numbers above, so they must be compared across scenarios only together.

## Pooled trip travel-time distribution

- Wasserstein distance: **1.65s**
- KS statistic: 0.0162 (p=0)

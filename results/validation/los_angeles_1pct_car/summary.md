# TorchDNL vs MATSim — DNL-fidelity validation summary

Seeds compared: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

## GEH statistic (replaces MSE/MAPE)

| series | n | mean | median | max | % GEH<5 | % GEH<10 |
|---|---|---|---|---|---|---|
| en_route | 2880 | 0.19 | 0.16 | 0.96 | 100.0% | 100.0% |
| departures | 2880 | 0.55 | 0.44 | 3.08 | 100.0% | 100.0% |
| arrivals | 2880 | 0.57 | 0.47 | 2.99 | 100.0% | 100.0% |
| link_volume | 23094540 | 0.01 | 0.00 | 2.45 | 100.0% | 100.0% |

## Envelope / self-consistency (en-route series)

- MATSim leave-one-out self-consistency (fraction of a held-out MATSim seed inside the envelope of the other MATSim seeds): **48.2%**
- TorchDNL fraction inside MATSim's inter-seed envelope: **24.7%**
- MATSim envelope width on loaded bins (median, relative to the median series): **0.1%**. An inclusion rate is only meaningful against this: the narrower the envelope, the harsher both numbers above, so they must be compared across scenarios only together.

## Pooled trip travel-time distribution

- Wasserstein distance: **85.85s**
- KS statistic: 0.0036 (p=1.75e-31)

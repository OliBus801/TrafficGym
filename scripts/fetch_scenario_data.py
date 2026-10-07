"""Download real-world MATSim scenario input data used by TrafficGym.

None of the real-world network/population data shipped with the paper's
results is authored by us: it is consumed as-is from public MATSim
open-scenario repositories (Sioux Falls from the MATSim ETH tutorial;
Berlin, Los Angeles, Leipzig, Vulkaneifel, Kelheim, Kyoto, Mexico City and
Ruhrgebiet from https://github.com/matsim-scenarios). To keep this
repository small, only small synthetic/toy scenarios are committed directly;
this script fetches the larger real-world scenarios on demand from their
original public hosts (the VSP TU-Berlin SVN server, which is where the
matsim-scenarios repos actually publish their large input files) into
`tamarl/data/scenarios/<name>/`.

License note: in every source repository the network/population XML files
are CC-BY 4.0, EXCEPT Kelheim, whose input data is CC BY-NC 4.0
(NonCommercial). See README/DATA docs for full attribution.

Usage:
    python scripts/fetch_scenario_data.py --scenario sioux_falls berlin
    python scripts/fetch_scenario_data.py --all
    python scripts/fetch_scenario_data.py --list
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "tamarl" / "data" / "scenarios"

SVN_BASE = "https://svn.vsp.tu-berlin.de/repos/public-svn/matsim/scenarios/countries"

# Each entry: source repo (for attribution), license, and the list of
# (download_url, local_filename) pairs to fetch. Local filenames drop the
# upstream `.gz` suffix once decompressed.
SCENARIOS = {
    "sioux_falls": {
        "repo": "matsim-eth/matsim-tutorial-ctu-2020",
        "license": "CC-BY 4.0",
        "files": [
            (
                "https://raw.githubusercontent.com/matsim-eth/matsim-tutorial-ctu-2020/"
                "235c0a7f4603937be0a2f1264f353ae745bb130f/scenarios/siouxfalls-2014/"
                "Siouxfalls_network_PT.xml",
                "Siouxfalls_network_PT.xml",
            ),
            (
                "https://raw.githubusercontent.com/matsim-eth/matsim-tutorial-ctu-2020/"
                "235c0a7f4603937be0a2f1264f353ae745bb130f/scenarios/siouxfalls-2014/"
                "Siouxfalls_population.xml.gz",
                "Siouxfalls_population.xml",
            ),
        ],
        "note": "Already shipped directly in the repository; fetching again just re-downloads it.",
    },
    "berlin": {
        "repo": "matsim-scenarios/matsim-berlin (tag v6.0)",
        "license": "CC-BY 4.0",
        "files": [
            (f"{SVN_BASE}/de/berlin/berlin-v6.0/input/berlin-v6.0-network-with-pt.xml.gz",
             "berlin-v6.0-network-with-pt.xml"),
            (f"{SVN_BASE}/de/berlin/berlin-v6.0/input/berlin-v6.0-facilities.xml.gz",
             "berlin-v6.0-facilities.xml"),
            (f"{SVN_BASE}/de/berlin/berlin-v6.0/input/berlin-v6.0-1pct.plans.xml.gz",
             "berlin-v6.0-1pct.plans.xml"),
            (f"{SVN_BASE}/de/berlin/berlin-v6.0/input/berlin-v6.0-10pct.plans.xml.gz",
             "berlin-v6.0-10pct.plans.xml"),
        ],
        "note": (
            "No official 100pct plans file is published upstream for v6.0; the paper's "
            "Berlin(100%) figures were produced from a scaled/merged population derived "
            "from these released fractions."
        ),
    },
    "los-angeles": {
        "repo": "matsim-scenarios/matsim-los-angeles (tag v1.0)",
        "license": "CC-BY 4.0",
        "files": [
            (f"{SVN_BASE}/us/los-angeles/los-angeles-v1.0/input/"
             "los-angeles-v1.0-network_2019-12-10.xml.gz",
             "los-angeles-v1.0-network_2019-12-10.xml"),
            (f"{SVN_BASE}/us/los-angeles/los-angeles-v1.0/input/"
             "los-angeles-v1.0-population-5pct_2020-03-17.xml.gz",
             "los-angeles-v1.0-population-5pct.xml"),
        ],
        "note": (
            "Capacities are left at their 100%-demand values (scale_factor=1.0), i.e. NOT "
            "rescaled to the 5% demand sample, matching the released configuration used in "
            "the paper. See tamarl/data/scenarios/berlin/clean_create_path.py for the "
            "car-only-leg filtering / free-flow route repair utility used to prepare a "
            "'-routed' population for the DNL validation appendix."
        ),
    },
    "leipzig": {
        "repo": "matsim-scenarios/matsim-leipzig (tag v1.3 / v1.3.1)",
        "license": "CC-BY 4.0",
        "files": [
            (f"{SVN_BASE}/de/leipzig/leipzig-v1.3/input/leipzig-v1.3-network-with-pt.xml.gz",
             "leipzig-v1.3-network-with-pt.xml"),
            (f"{SVN_BASE}/de/leipzig/leipzig-v1.3/input/leipzig-v1.3.1-10pct.plans.xml.gz",
             "leipzig-v1.3.1-10pct.plans.xml"),
        ],
        "note": "Fully integrated scenario (config_Leipzig.json).",
    },
    "vulkaneifel": {
        "repo": "matsim-scenarios/matsim-vulkaneifel (tag v1.2)",
        "license": "CC-BY 4.0",
        "files": [
            (f"{SVN_BASE}/de/vulkaneifel/v1.2/input/vulkaneifel-v1.2-network-with-pt.xml.gz",
             "vulkaneifel-v1.2-network-with-pt.xml"),
            (f"{SVN_BASE}/de/vulkaneifel/v1.2/input/vulkaneifel-v1.2-25pct.plans.xml.gz",
             "vulkaneifel-v1.2-25pct.plans.xml"),
        ],
        "note": "Fully integrated scenario (config_Vulkaneifel.json).",
    },
    "kelheim": {
        "repo": "matsim-scenarios/matsim-kelheim (tag v3.0)",
        "license": "CC BY-NC 4.0 (NonCommercial)",
        "files": [
            (f"{SVN_BASE}/de/kelheim/kelheim-v3.0/input/kelheim-v3.0-network-with-pt.xml.gz",
             "kelheim-v3.0-network-with-pt.xml"),
            (f"{SVN_BASE}/de/kelheim/kelheim-v3.0/input/kelheim-v3.0-25pct-plans.xml.gz",
             "kelheim-v3.0-25pct-plans.xml"),
        ],
        "note": "Visualization-only in the paper (unrouted population); NonCommercial license.",
    },
    "kyoto": {
        "repo": "matsim-scenarios/matsim-kyoto (no tagged release; pinned commit)",
        "license": "CC-BY 4.0",
        "files": [
            (f"{SVN_BASE}/jp/kyoto/input/v1.0/kyoto-v1.0-network-with-pt.xml.gz",
             "kyoto-v1.0-network-with-pt.xml"),
            (f"{SVN_BASE}/jp/kyoto/input/v1.0/kyoto-v1.0-10pct.plans-initial.xml.gz",
             "kyoto-v1.0-10pct.plans-initial.xml"),
        ],
        "note": "Visualization-only in the paper (unrouted population).",
    },
    "mexico-city": {
        "repo": "matsim-scenarios/matsim-mexico-city (tag v1.0)",
        "license": "CC-BY 4.0",
        "files": [
            (f"{SVN_BASE}/mx/mexico-city/mexico-city-v1.0/input/"
             "mexico-city-v1.0-network-with-pt.xml.gz",
             "mexico-city-v1.0-network-with-pt.xml"),
            (f"{SVN_BASE}/mx/mexico-city/mexico-city-v1.0/input/"
             "mexico-city-v1.0-1pct.input.plans.xml.gz",
             "mexico-city-v1.0-1pct.plans.xml"),
        ],
        "note": "Visualization-only in the paper (unrouted population).",
    },
    "ruhrgebiet": {
        "repo": "matsim-scenarios/matsim-ruhrgebiet (tag v1.1)",
        "license": "CC-BY 4.0",
        "files": [
            (f"{SVN_BASE}/de/ruhrgebiet/ruhrgebiet-v1.1-1pct/input/"
             "ruhrgebiet-v1.1.network.xml.gz",
             "ruhrgebiet-v1.1-network.xml"),
            (f"{SVN_BASE}/de/ruhrgebiet/ruhrgebiet-v1.1-1pct/input/"
             "ruhrgebiet-v1.1-1pct.plans.xml.gz",
             "ruhrgebiet-v1.1-1pct-plans.xml"),
        ],
        "note": "Visualization-only in the paper (unrouted population).",
    },
}


def _download(url: str, dest: Path) -> None:
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    if url.endswith(".gz"):
        dest.write_bytes(gzip.decompress(data))
    else:
        dest.write_bytes(data)


def fetch_scenario(name: str, force: bool = False) -> None:
    spec = SCENARIOS[name]
    out_dir = SCENARIOS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] source: {spec['repo']} ({spec['license']})")
    for url, filename in spec["files"]:
        dest = out_dir / filename
        if dest.exists() and not force:
            print(f"  skip (exists): {dest}")
            continue
        _download(url, dest)
        print(f"  saved: {dest}")
    if spec.get("note"):
        print(f"  note: {spec['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", nargs="+", choices=sorted(SCENARIOS), help="Scenario(s) to fetch"
    )
    parser.add_argument("--all", action="store_true", help="Fetch every real-world scenario")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    if args.list or not (args.scenario or args.all):
        print("Available scenarios:")
        for name, spec in SCENARIOS.items():
            print(f"  {name:15s} {spec['repo']}  [{spec['license']}]")
        if not (args.scenario or args.all):
            print("\nPass --scenario NAME [NAME ...] or --all to download.")
        return

    names = list(SCENARIOS) if args.all else args.scenario
    for name in names:
        fetch_scenario(name, force=args.force)


if __name__ == "__main__":
    main()

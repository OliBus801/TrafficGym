"""Scale a MATSim population XML by an integer factor, for controlled
population-scale experiments.

Replaces the Sioux-Falls-specific `tamarl/data/scenarios/sioux_falls/
scale_population.py`: works on any scenario, and clones by streaming over the
source text instead of building an ElementTree, because at scale=16 on Sioux
Falls the ElementTree route means ~1M `deepcopy` calls and a ~512 MB document
held in RAM.

Two cloning modes:

  exact (default)
      Every person is duplicated verbatim, S times, with a unique id. OD pairs,
      activity chains and departure times are bit-identical across clones, so
      the demand profile is *literally* unchanged and population is the only
      manipulated variable. This is the clean control.

  jitter
      Same, but every `start_time`/`end_time` in a cloned plan is shifted by one
      per-clone delta drawn uniformly from +/- `--jitter` seconds. Activity
      durations are preserved (the whole chain shifts together). Use as a
      robustness check: exact cloning puts S vehicles on the same OD at the same
      second, and this shows whether that atomicity matters.

Usage:
    python scripts/scale_population.py \\
        --input tamarl/data/scenarios/sioux_falls/Siouxfalls_population.xml \\
        --scale 16 --tag S16
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

PERSON_RE = re.compile(r"<person\b.*?</person>", re.DOTALL)
PERSON_ID_RE = re.compile(r'(<person\b[^>]*?\bid=")([^"]*)(")')
TIME_RE = re.compile(r'\b(start_time|end_time)="(\d+):(\d\d):(\d\d)"')

HEADER_XMLDECL = '<?xml version="1.0" encoding="utf-8"?>\n'
FOOTER = "</population>\n"


def source_doctype(path, default_version=5):
    """Return the source file's DOCTYPE line, so the output declares the same schema.

    Hard-coding population_v5 silently produces files MATSim refuses: Berlin v6.0
    plans carry <attributes> blocks that only population_v6 defines. TorchDNL's
    iterparse ignores the DTD entirely, so the mismatch shows up only once the
    other engine reads the file.
    """
    with open(path, errors="replace") as f:
        for _ in range(5):
            line = f.readline()
            if "<!DOCTYPE" in line:
                return line if line.endswith("\n") else line + "\n"
    return (
        '<!DOCTYPE population SYSTEM '
        f'"http://www.matsim.org/files/dtd/population_v{default_version}.dtd">\n'
    )



def shift_times(block: str, delta: int) -> str:
    """Shift every start_time/end_time in `block` by `delta` seconds, clamped at 0."""

    def sub(m: re.Match) -> str:
        attr, h, mi, s = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        t = max(0, h * 3600 + mi * 60 + s + delta)
        return f'{attr}="{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"'

    return TIME_RE.sub(sub, block)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Source population XML")
    p.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Loading factor (>0). Integer values clone the whole population; "
        "fractional values subsample (<1) or clone plus subsample (>1), so that "
        "the output holds round(n * scale) persons.",
    )
    p.add_argument("--output", default=None, help="Output path (default: alongside input, named by --tag)")
    p.add_argument(
        "--tag",
        default=None,
        help="Filename token used to select this file via --population (default: S<scale>)",
    )
    p.add_argument(
        "--jitter",
        type=int,
        default=0,
        help="If >0, shift each clone's whole activity chain by a uniform delta in +/- this many seconds",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --jitter")
    args = p.parse_args()

    if args.scale <= 0:
        print("ABORT: --scale must be > 0", file=sys.stderr)
        return 1

    src = Path(args.input)
    if not src.exists():
        print(f"ABORT: {src} not found", file=sys.stderr)
        return 1

    # "." would be split by scenario_loader's filename tokeniser, so a fractional
    # loading is spelled S1p5 rather than S1.5.
    tag = args.tag or ("S" + f"{args.scale:g}".replace(".", "p"))
    if args.jitter:
        tag = f"{tag}j{args.jitter}"
    out = Path(args.output) if args.output else src.parent / f"{tag}_population.xml"

    text = src.read_text(encoding="utf-8")
    blocks = PERSON_RE.findall(text)
    if not blocks:
        print(f"ABORT: no <person> elements found in {src}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    n_in = len(blocks)
    n_target = round(n_in * args.scale)
    if n_target < 1:
        print(f"ABORT: scale {args.scale} on {n_in} persons rounds to 0", file=sys.stderr)
        return 1

    # Emit (person index, clone index) pairs: whole passes over the population for
    # the integer part, then a random subsample for the remainder, so the output
    # holds exactly n_target persons. Subsampling by person keeps every emitted
    # activity chain intact.
    full_passes, remainder = divmod(n_target, n_in)
    emit: list[tuple[int, int]] = [(i, c) for c in range(full_passes) for i in range(n_in)]
    if remainder:
        emit += [(i, full_passes) for i in sorted(rng.sample(range(n_in), remainder))]

    n_written = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write(HEADER_XMLDECL + source_doctype(args.input) + "<population>\n")
        for i, c in emit:
            block = blocks[i]
            if c == 0:
                clone = block
            else:
                m = PERSON_ID_RE.search(block)
                orig_id = m.group(2) if m else None
                clone = PERSON_ID_RE.sub(rf"\g<1>\g<2>_c{c}\g<3>", block)
                # Routes may name the vehicle they use (Berlin: vehicleRefId=
                # "<person>_car"). Renaming only the person would leave every clone
                # pointing at the ORIGINAL person's vehicle, and MATSim aborts with
                # "Could not get the requested vehicle with ID=..." as soon as the
                # scale factor exceeds 1. TorchDNL ignores the attribute entirely,
                # so this breaks one engine and not the other.
                if orig_id:
                    clone = clone.replace(
                        f'vehicleRefId="{orig_id}_', f'vehicleRefId="{orig_id}_c{c}_'
                    )
            if c > 0 and args.jitter:
                clone = shift_times(clone, rng.randint(-args.jitter, args.jitter))
            f.write(clone)
            f.write("\n")
            n_written += 1
        f.write(FOOTER)

    size_mb = out.stat().st_size / 1e6
    print(
        f"{src.name}: {len(blocks)} persons x{args.scale}"
        f"{f' (jitter +/-{args.jitter}s)' if args.jitter else ''}"
        f" -> {n_written} persons, {size_mb:.1f} MB"
    )
    print(f"wrote {out}")
    print(f"select it with: --population {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

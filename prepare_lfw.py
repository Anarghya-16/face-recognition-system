"""Build a free, real-world evaluation dataset from Labeled Faces in the Wild.

LFW is a public academic benchmark, so the whole evaluation stays at zero cost.
Identities with enough images become the *known* set; a disjoint group of other
identities becomes the *unknown* impostor set that the system must reject.

    python prepare_lfw.py --known 20 --unknown 20 --min-faces 5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download and lay out an LFW subset.")
    ap.add_argument("--known", type=int, default=20, help="Number of enrolled identities")
    ap.add_argument("--unknown", type=int, default=20, help="Number of impostor identities")
    ap.add_argument("--min-faces", type=int, default=5,
                    help="Minimum images per identity for the known set")
    ap.add_argument("--out", default="data", help="Output root")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        raise SystemExit("pip install scikit-learn Pillow to use this script.")
    from PIL import Image

    rng = np.random.default_rng(args.seed)

    print("Fetching LFW (first run downloads ~200 MB) ...")
    known_src = fetch_lfw_people(min_faces_per_person=args.min_faces,
                                 color=True, resize=1.0, funneled=True)
    names = list(known_src.target_names)
    order = rng.permutation(len(names))
    chosen_known = set(order[: args.known].tolist())
    chosen_unknown = set(order[args.known: args.known + args.unknown].tolist())

    out_known = Path(args.out) / "enroll"
    out_unknown = Path(args.out) / "unknown"
    for d in (out_known, out_unknown):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    counts = {"known": 0, "unknown": 0}
    for img, target in zip(known_src.images, known_src.target):
        if target in chosen_known:
            root, key = out_known, "known"
        elif target in chosen_unknown:
            root, key = out_unknown, "unknown"
        else:
            continue
        person = names[target].replace(" ", "_")
        person_dir = root / person
        person_dir.mkdir(exist_ok=True)
        idx = len(list(person_dir.glob("*.jpg")))
        arr = (img * 255).astype("uint8") if img.max() <= 1.0 else img.astype("uint8")
        Image.fromarray(arr).save(person_dir / f"{idx:03d}.jpg", quality=95)
        counts[key] += 1

    print(f"Known   : {len(list(out_known.iterdir()))} identities, "
          f"{counts['known']} images -> {out_known}")
    print(f"Unknown : {len(list(out_unknown.iterdir()))} identities, "
          f"{counts['unknown']} images -> {out_unknown}")
    print("\nNext:\n  python enroll.py --data data/enroll\n"
          "  python evaluate.py --data data/enroll --unknown-data data/unknown")


if __name__ == "__main__":
    main()

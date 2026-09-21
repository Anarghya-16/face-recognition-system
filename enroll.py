"""Enrol people into the face gallery.

Usage
-----
    # enrol every person in data/enroll/<name>/*.jpg
    python enroll.py --data data/enroll --gallery models/gallery.npz

    # add or update one person only
    python enroll.py --person "Anarghya" --images data/enroll/Anarghya --append
"""

from __future__ import annotations

import argparse
from pathlib import Path

from face_id import FaceEngine, Gallery
from face_id.utils import list_images, load_identity_folders


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Enrol faces into the gallery.")
    ap.add_argument("--data", default="data/enroll",
                    help="Dataset root laid out as <root>/<person>/<image>.jpg")
    ap.add_argument("--person", help="Enrol a single named person instead of a whole root")
    ap.add_argument("--images", help="Folder of images for --person")
    ap.add_argument("--gallery", default="models/gallery.npz", help="Output .npz path")
    ap.add_argument("--append", action="store_true",
                    help="Add to an existing gallery instead of rebuilding it")
    ap.add_argument("--min-det-score", type=float, default=0.60,
                    help="Reject enrolment images whose detection confidence is lower")
    ap.add_argument("--ctx-id", type=int, default=-1, help="-1 for CPU, 0 for first GPU")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    engine = FaceEngine(ctx_id=args.ctx_id)

    gallery = Gallery.load(args.gallery) if (args.append and Path(args.gallery).exists()) \
        else Gallery()

    if args.person:
        if not args.images:
            raise SystemExit("--person requires --images <folder>")
        people = {args.person: list_images(args.images)}
        gallery.remove(args.person)          # re-enrolling replaces old templates
    else:
        people = load_identity_folders(args.data)

    enrolled, skipped = 0, 0
    for name, paths in people.items():
        accepted = 0
        for p in paths:
            try:
                faces = engine.detect_path(str(p))
            except Exception as exc:                      # unreadable / corrupt file
                print(f"  [skip] {p.name}: {exc}")
                skipped += 1
                continue
            if not faces:
                print(f"  [skip] {p.name}: no face detected")
                skipped += 1
                continue
            if len(faces) > 1:
                print(f"  [warn] {p.name}: {len(faces)} faces, using the largest")
            face = faces[0]
            if face.det_score < args.min_det_score:
                print(f"  [skip] {p.name}: low detector confidence {face.det_score:.2f}")
                skipped += 1
                continue
            gallery.add(face.embedding, name, source=str(p))
            accepted += 1
            enrolled += 1
        print(f"{name}: {accepted}/{len(paths)} images enrolled")
        if accepted == 0:
            print(f"  [warn] {name} has no usable template and cannot be recognised")

    gallery.save(args.gallery)
    print(
        f"\nGallery saved to {args.gallery} -- "
        f"{len(gallery)} templates across {len(gallery.identities)} identities "
        f"({skipped} images skipped)."
    )


if __name__ == "__main__":
    main()

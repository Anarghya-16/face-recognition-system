"""Identify faces in an image, a folder of images, or a webcam stream.

Usage
-----
    python identify.py --image data/test/query.jpg
    python identify.py --folder data/test --json results/predictions.json
    python identify.py --webcam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from face_id import FaceEngine, Gallery
from face_id.utils import list_images


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Identify faces against the gallery.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Single image to identify")
    src.add_argument("--folder", help="Folder of images to identify")
    src.add_argument("--webcam", action="store_true", help="Live identification from camera")
    ap.add_argument("--gallery", default="models/gallery.npz")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="Cosine similarity below which a face is declared unknown")
    ap.add_argument("--aggregate", choices=["max", "mean"], default="max")
    ap.add_argument("--save-dir", default="results/annotated",
                    help="Where annotated images are written")
    ap.add_argument("--json", help="Optional path to dump predictions as JSON")
    ap.add_argument("--ctx-id", type=int, default=-1)
    return ap.parse_args()


def run_image(engine: FaceEngine, gallery: Gallery, path: Path,
              args: argparse.Namespace) -> dict:
    img = engine.read_image(str(path))
    faces = engine.detect(img)
    record = {"image": str(path), "num_faces": len(faces), "faces": []}
    for face in faces:
        match = gallery.identify(face.embedding, args.threshold, args.aggregate)
        record["faces"].append({
            "bbox": [round(float(v), 1) for v in face.bbox],
            "det_score": round(face.det_score, 3),
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in match.to_dict().items()},
        })
        engine.annotate(img, face, match.label, match.score, match.known)
        print(f"{path.name}: {match.label} (score={match.score:.3f}, "
              f"nearest={match.best_label})")
    if not faces:
        print(f"{path.name}: no face detected")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / path.name), img)
    return record


def run_webcam(engine: FaceEngine, gallery: Gallery, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for face in engine.detect(frame):
            match = gallery.identify(face.embedding, args.threshold, args.aggregate)
            engine.annotate(frame, face, match.label, match.score, match.known)
        cv2.imshow("Face Identification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    engine = FaceEngine(ctx_id=args.ctx_id)
    gallery = Gallery.load(args.gallery)
    print(f"Loaded {len(gallery)} templates for {len(gallery.identities)} identities "
          f"at threshold {args.threshold}.\n")

    if args.webcam:
        run_webcam(engine, gallery, args)
        return

    paths = [Path(args.image)] if args.image else list_images(args.folder)
    if not paths:
        raise SystemExit("No images found.")
    records = [run_image(engine, gallery, p, args) for p in paths]

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2))
        print(f"\nPredictions written to {out}")


if __name__ == "__main__":
    main()

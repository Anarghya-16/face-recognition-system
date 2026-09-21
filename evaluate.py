"""Open-set evaluation of the face identification system.

Protocol
--------
Identities are split into two disjoint groups:

* **Known**   -- the first `--gallery-per-id` images are enrolled, the rest
                 become probes whose correct answer is their own name.
* **Unknown** -- never enrolled; every image becomes a probe whose correct
                 answer is the literal label `unknown`.

Reported metrics
----------------
* Verification view : ROC-AUC and EER over genuine vs impostor similarity scores.
* Identification view (per threshold):
    - Rank-1 accuracy (closed set, thresholding ignored)
    - TAR / correct identification rate on known probes
    - FAR: unknown probes wrongly accepted
    - Unknown rejection rate
    - Overall open-set accuracy and macro-F1

Usage
-----
    python evaluate.py --data data/enroll --unknown-data data/unknown
    python evaluate.py --data data/dataset --unknown-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from face_id import FaceEngine, Gallery
from face_id.gallery import UNKNOWN_LABEL
from face_id.utils import load_identity_folders


# --------------------------------------------------------------------- args --
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate the identification system.")
    ap.add_argument("--data", default="data/enroll", help="Known-identity dataset root")
    ap.add_argument("--unknown-data", help="Optional folder of never-enrolled identities")
    ap.add_argument("--unknown-ratio", type=float, default=0.0,
                    help="If no --unknown-data, hold out this fraction of identities "
                         "from --data as unknown impostors")
    ap.add_argument("--gallery-per-id", type=int, default=2,
                    help="Images per known identity used for enrolment")
    ap.add_argument("--aggregate", choices=["max", "mean"], default="max")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="Operating threshold reported in detail")
    ap.add_argument("--out", default="results", help="Directory for metrics and plots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx-id", type=int, default=-1)
    return ap.parse_args()


# ------------------------------------------------------------------ helpers --
def embed_folder_dataset(engine: FaceEngine, people: Dict[str, List[Path]]
                         ) -> Dict[str, List[np.ndarray]]:
    """Detect the largest face per image and return embeddings grouped by identity."""
    out: Dict[str, List[np.ndarray]] = {}
    failures = 0
    for name, paths in people.items():
        embs = []
        for p in paths:
            try:
                face = engine.largest_face_path(str(p))
            except Exception:
                face = None
            if face is None:
                failures += 1
                continue
            embs.append(face.embedding)
        if embs:
            out[name] = embs
    if failures:
        print(f"[note] {failures} images produced no detection and were dropped.")
    return out


def eer_from_scores(genuine: np.ndarray, impostor: np.ndarray) -> Tuple[float, float]:
    """Equal Error Rate and the threshold at which it occurs."""
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan"), float("nan")
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    best_gap, eer, eer_thr = np.inf, float("nan"), float("nan")
    for t in thresholds:
        frr = float(np.mean(genuine < t))     # genuine pairs rejected
        far = float(np.mean(impostor >= t))   # impostor pairs accepted
        gap = abs(frr - far)
        if gap < best_gap:
            best_gap, eer, eer_thr = gap, (frr + far) / 2.0, float(t)
    return eer, eer_thr


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic (no scikit-learn dependency required)."""
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan")
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(genuine), len(impostor)
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# --------------------------------------------------------------------- main --
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = FaceEngine(ctx_id=args.ctx_id)

    print("Embedding the known-identity dataset ...")
    known_embs = embed_folder_dataset(engine, load_identity_folders(args.data))

    unknown_embs: Dict[str, List[np.ndarray]] = {}
    if args.unknown_data:
        print("Embedding the unknown-identity dataset ...")
        unknown_embs = embed_folder_dataset(engine, load_identity_folders(args.unknown_data))
    elif args.unknown_ratio > 0:
        names = sorted(known_embs)
        random.shuffle(names)
        n_unknown = max(1, int(round(len(names) * args.unknown_ratio)))
        for name in names[:n_unknown]:
            unknown_embs[name] = known_embs.pop(name)
        print(f"Held out {len(unknown_embs)} identities as unknown impostors.")

    # -- build the gallery and the probe set -------------------------------
    gallery = Gallery()
    probes: List[Tuple[np.ndarray, str]] = []       # (embedding, ground-truth label)
    usable_ids = 0
    for name, embs in known_embs.items():
        if len(embs) < args.gallery_per_id + 1:
            print(f"[warn] {name}: only {len(embs)} usable images; "
                  f"needs {args.gallery_per_id + 1} to contribute a probe.")
        for e in embs[:args.gallery_per_id]:
            gallery.add(e, name, source="eval")
        for e in embs[args.gallery_per_id:]:
            probes.append((e, name))
        usable_ids += 1
    for name, embs in unknown_embs.items():
        for e in embs:
            probes.append((e, UNKNOWN_LABEL))

    if len(gallery) == 0 or not probes:
        raise SystemExit("Not enough data: need enrolled templates and probe images.")

    # -- score every probe against the gallery once ------------------------
    genuine, impostor = [], []
    rows = []
    for emb, truth in probes:
        scores = gallery.per_identity_scores(emb, aggregate=args.aggregate)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = ranked[0]
        rows.append({"truth": truth, "best_label": best_label, "best_score": best_score})
        if truth == UNKNOWN_LABEL:
            impostor.append(best_score)
        else:
            genuine.append(scores.get(truth, -1.0))
            for lab, s in scores.items():
                if lab != truth:
                    impostor.append(s)

    genuine_a, impostor_a = np.array(genuine), np.array(impostor)
    auc = roc_auc(genuine_a, impostor_a)
    eer, eer_thr = eer_from_scores(genuine_a, impostor_a)

    known_rows = [r for r in rows if r["truth"] != UNKNOWN_LABEL]
    rank1 = (np.mean([r["best_label"] == r["truth"] for r in known_rows])
             if known_rows else float("nan"))

    # -- threshold sweep ---------------------------------------------------
    sweep = []
    for t in np.round(np.arange(0.10, 0.81, 0.01), 2):
        y_true, y_pred = [], []
        for r in rows:
            pred = r["best_label"] if r["best_score"] >= t else UNKNOWN_LABEL
            y_true.append(r["truth"])
            y_pred.append(pred)
        known_mask = [t_ != UNKNOWN_LABEL for t_ in y_true]
        unk_mask = [not m for m in known_mask]
        cir = (np.mean([p == t_ for p, t_, m in zip(y_pred, y_true, known_mask) if m])
               if any(known_mask) else float("nan"))
        far = (np.mean([p != UNKNOWN_LABEL for p, m in zip(y_pred, unk_mask) if m])
               if any(unk_mask) else float("nan"))
        sweep.append({
            "threshold": float(t),
            "correct_identification_rate": float(cir),
            "false_accept_rate_unknown": float(far),
            "unknown_rejection_rate": float(1 - far) if not np.isnan(far) else float("nan"),
            "overall_accuracy": float(np.mean([p == t_ for p, t_ in zip(y_pred, y_true)])),
            "macro_f1": macro_f1(y_true, y_pred),
        })

    best = max(sweep, key=lambda d: d["macro_f1"])
    chosen = min(sweep, key=lambda d: abs(d["threshold"] - args.threshold))

    metrics = {
        "config": {
            "model": engine.model_name,
            "aggregate": args.aggregate,
            "gallery_images_per_identity": args.gallery_per_id,
            "enrolled_identities": len(gallery.identities),
            "enrolled_templates": len(gallery),
            "known_probes": len(known_rows),
            "unknown_probes": len(rows) - len(known_rows),
        },
        "verification": {
            "roc_auc": auc,
            "eer": eer,
            "eer_threshold": eer_thr,
            "genuine_mean": float(genuine_a.mean()) if len(genuine_a) else None,
            "genuine_std": float(genuine_a.std()) if len(genuine_a) else None,
            "impostor_mean": float(impostor_a.mean()) if len(impostor_a) else None,
            "impostor_std": float(impostor_a.std()) if len(impostor_a) else None,
        },
        "identification": {
            "rank1_closed_set": float(rank1),
            "at_chosen_threshold": chosen,
            "best_by_macro_f1": best,
        },
        "threshold_sweep": sweep,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # -- plots (optional; matplotlib may be absent) ------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(impostor_a, bins=40, alpha=0.6, label="impostor", density=True)
        ax.hist(genuine_a, bins=40, alpha=0.6, label="genuine", density=True)
        ax.axvline(args.threshold, ls="--", c="k", label=f"threshold={args.threshold}")
        ax.set_xlabel("cosine similarity"); ax.set_ylabel("density")
        ax.set_title("Genuine vs impostor score distributions"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "score_distributions.png", dpi=150)

        ts = [d["threshold"] for d in sweep]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts, [d["correct_identification_rate"] for d in sweep], label="correct ID rate")
        ax.plot(ts, [d["unknown_rejection_rate"] for d in sweep], label="unknown rejection")
        ax.plot(ts, [d["macro_f1"] for d in sweep], label="macro-F1")
        ax.axvline(best["threshold"], ls="--", c="k",
                   label=f"best F1 @ {best['threshold']:.2f}")
        ax.set_xlabel("threshold"); ax.set_ylabel("rate"); ax.set_ylim(0, 1.02)
        ax.set_title("Threshold sweep"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "threshold_sweep.png", dpi=150)
        print(f"Plots written to {out_dir}")
    except Exception as exc:
        print(f"[note] plots skipped ({exc})")

    # -- console summary ---------------------------------------------------
    print("\n=== Evaluation summary ===")
    print(f"Identities enrolled      : {len(gallery.identities)} "
          f"({len(gallery)} templates)")
    print(f"Probes                   : {len(known_rows)} known, "
          f"{len(rows) - len(known_rows)} unknown")
    print(f"Rank-1 (closed set)      : {rank1:.3f}")
    print(f"ROC-AUC / EER            : {auc:.4f} / {eer:.4f} (at {eer_thr:.3f})")
    print(f"Genuine   mean +/- std   : {genuine_a.mean():.3f} +/- {genuine_a.std():.3f}")
    print(f"Impostor  mean +/- std   : {impostor_a.mean():.3f} +/- {impostor_a.std():.3f}")
    print(f"\nAt threshold {chosen['threshold']:.2f}: "
          f"correct ID {chosen['correct_identification_rate']:.3f}, "
          f"unknown rejection {chosen['unknown_rejection_rate']:.3f}, "
          f"macro-F1 {chosen['macro_f1']:.3f}")
    print(f"Best macro-F1 threshold  : {best['threshold']:.2f} "
          f"(F1 {best['macro_f1']:.3f})")
    print(f"\nFull metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

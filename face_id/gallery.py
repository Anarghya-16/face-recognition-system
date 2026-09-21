"""Enrolment database and similarity-based matching.

The gallery keeps every enrolled template (one 512-d embedding per enrolment
image) rather than only a per-person average. At query time the score for a
person is the aggregate of the query's cosine similarity against that person's
templates -- `max` by default, which tolerates pose/lighting variation better
than a centroid, and `mean`, which is more robust to a single bad enrolment
image. Both are available so the trade-off can be shown empirically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

UNKNOWN_LABEL = "unknown"


@dataclass
class Match:
    label: str            # predicted identity, or "unknown"
    score: float          # cosine similarity of the best-scoring identity
    known: bool           # True if score >= threshold
    best_label: str       # nearest identity regardless of threshold
    runner_up: Optional[str] = None
    runner_up_score: Optional[float] = None

    @property
    def margin(self) -> Optional[float]:
        """Gap between the best and second-best identity (a confidence cue)."""
        if self.runner_up_score is None:
            return None
        return self.score - self.runner_up_score

    def to_dict(self) -> dict:
        d = asdict(self)
        d["margin"] = self.margin
        return d


class Gallery:
    """An in-memory template store persisted as a single .npz file."""

    def __init__(self, embeddings: Optional[np.ndarray] = None,
                 labels: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None) -> None:
        self.embeddings = (
            np.zeros((0, 512), dtype=np.float32) if embeddings is None
            else np.asarray(embeddings, dtype=np.float32)
        )
        self.labels: List[str] = list(labels or [])
        self.sources: List[str] = list(sources or [])

    # ------------------------------------------------------------ mutation --
    def add(self, embedding: np.ndarray, label: str, source: str = "") -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self.embeddings = np.vstack([self.embeddings, emb]) if len(self.embeddings) else emb
        self.labels.append(label)
        self.sources.append(source)

    def remove(self, label: str) -> int:
        """Delete every template of one identity. Returns the number removed."""
        keep = [i for i, l in enumerate(self.labels) if l != label]
        removed = len(self.labels) - len(keep)
        self.embeddings = self.embeddings[keep] if keep else np.zeros((0, 512), np.float32)
        self.labels = [self.labels[i] for i in keep]
        self.sources = [self.sources[i] for i in keep]
        return removed

    # ------------------------------------------------------------ querying --
    @property
    def identities(self) -> List[str]:
        return sorted(set(self.labels))

    def __len__(self) -> int:
        return len(self.labels)

    def per_identity_scores(self, embedding: np.ndarray,
                            aggregate: str = "max") -> Dict[str, float]:
        """Cosine similarity of `embedding` against every enrolled identity."""
        if len(self) == 0:
            return {}
        q = np.asarray(embedding, dtype=np.float32).ravel()
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        sims = self.embeddings @ q          # both sides are L2-normalised
        scores: Dict[str, List[float]] = {}
        for label, s in zip(self.labels, sims):
            scores.setdefault(label, []).append(float(s))
        if aggregate == "mean":
            return {k: float(np.mean(v)) for k, v in scores.items()}
        return {k: float(np.max(v)) for k, v in scores.items()}

    def identify(self, embedding: np.ndarray, threshold: float = 0.35,
                 aggregate: str = "max") -> Match:
        """Nearest-identity search with explicit rejection below `threshold`."""
        scores = self.per_identity_scores(embedding, aggregate=aggregate)
        if not scores:
            return Match(UNKNOWN_LABEL, 0.0, False, UNKNOWN_LABEL)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = ranked[0]
        runner_up, runner_up_score = (ranked[1] if len(ranked) > 1 else (None, None))
        known = best_score >= threshold
        return Match(
            label=best_label if known else UNKNOWN_LABEL,
            score=best_score,
            known=known,
            best_label=best_label,
            runner_up=runner_up,
            runner_up_score=runner_up_score,
        )

    # --------------------------------------------------------- persistence --
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            labels=np.array(self.labels, dtype=object),
            sources=np.array(self.sources, dtype=object),
        )
        meta = {
            "num_templates": len(self),
            "num_identities": len(self.identities),
            "identities": {i: self.labels.count(i) for i in self.identities},
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No gallery at {path}. Run `python enroll.py` first."
            )
        data = np.load(path, allow_pickle=True)
        return cls(
            embeddings=data["embeddings"],
            labels=[str(x) for x in data["labels"]],
            sources=[str(x) for x in data["sources"]],
        )

"""Face detection and embedding extraction.

A thin, explicit wrapper around InsightFace's `buffalo_l` model pack:

* Detector  : SCRFD-10GF  (returns bbox, det_score, 5 facial landmarks)
* Recogniser: ArcFace w600k_r50 (ResNet-50 trained with ArcFace margin loss,
              512-d embedding, already L2-normalised by InsightFace)

Both models are ONNX and run on CPU through onnxruntime, so the whole system
has zero licensing or infrastructure cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class DetectedFace:
    """One detected face with its aligned-crop embedding."""

    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    det_score: float          # detector confidence in [0, 1]
    embedding: np.ndarray     # (512,) float32, L2-normalised
    kps: Optional[np.ndarray] = None   # (5, 2) landmarks

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


class FaceEngine:
    """Detect faces in an image and return normalised ArcFace embeddings."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple = (640, 640),
        det_thresh: float = 0.50,
        ctx_id: int = -1,          # -1 = CPU, >=0 = GPU device id
    ) -> None:
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )
        # Only the detection and recognition modules are needed; skipping the
        # landmark/attribute models keeps inference roughly 2x faster on CPU.
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        self.det_thresh = det_thresh
        self.model_name = model_name

    # ------------------------------------------------------------------ IO --
    @staticmethod
    def read_image(path: str) -> np.ndarray:
        """Read an image as BGR uint8 (InsightFace expects BGR, like OpenCV)."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img

    # ----------------------------------------------------------- inference --
    def detect(self, img: np.ndarray) -> List[DetectedFace]:
        """Return every detected face, sorted by bounding-box area (largest first)."""
        faces = self.app.get(img)
        out: List[DetectedFace] = []
        for f in faces:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            # Defensive re-normalisation: guarantees cosine similarity == dot product.
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            out.append(
                DetectedFace(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    det_score=float(f.det_score),
                    embedding=emb,
                    kps=np.asarray(f.kps, dtype=np.float32) if f.kps is not None else None,
                )
            )
        out.sort(key=lambda d: d.area, reverse=True)
        return out

    def detect_path(self, path: str) -> List[DetectedFace]:
        return self.detect(self.read_image(path))

    def largest_face(self, img: np.ndarray) -> Optional[DetectedFace]:
        """Convenience helper for enrolment, where one face per image is assumed."""
        faces = self.detect(img)
        return faces[0] if faces else None

    def largest_face_path(self, path: str) -> Optional[DetectedFace]:
        return self.largest_face(self.read_image(path))

    # ------------------------------------------------------------- drawing --
    @staticmethod
    def annotate(
        img: np.ndarray,
        face: DetectedFace,
        label: str,
        score: float,
        known: bool,
    ) -> np.ndarray:
        """Draw a labelled box; green for an accepted match, red for `unknown`."""
        colour = (0, 180, 0) if known else (0, 0, 220)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        text = f"{label} {score:.3f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(
            img, text, (x1 + 3, max(th, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return img

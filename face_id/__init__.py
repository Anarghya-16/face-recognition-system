"""Face Recognition Identification System.

Pipeline: detection (SCRFD) -> alignment -> embedding (ArcFace) ->
cosine similarity matching against an enrolled gallery -> threshold-based
acceptance with an explicit `unknown` decision.
"""

from .engine import FaceEngine, DetectedFace
from .gallery import Gallery, Match

__all__ = ["FaceEngine", "DetectedFace", "Gallery", "Match"]
__version__ = "1.0.0"

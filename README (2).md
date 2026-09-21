# Face Recognition Identification System

A face identification system that **enrolls** people into a database and **identifies** new faces by matching them against the enrolled identities. Faces that don't match anyone well enough are rejected as **unknown**.

Built for the AI/ML Intern assignment at Code Nimbus Solutions using only free, open-source tools (₹0 / $0 spend).

- **Colab notebook (full run with outputs):** https://colab.research.google.com/drive/1hOi5wZjDx4D0evW1W7VS1qL6WhtEcKKC
- **Code:** [`face_id/`](face_id/)

## Pipeline

```
Image → Face detection → Landmark alignment → ArcFace embedding → Cosine similarity vs. enrolled DB → Threshold → Identity or "unknown"
```

1. **Enroll:** detect the face, compute its 512-d embedding, and store it under the person's name.
2. **Identify:** detect faces in a new image, embed each one, compute cosine similarity against every enrolled embedding, and take the best match.
3. **Reject unknowns:** if the best similarity is below the threshold, the face is labelled `unknown` instead of being forced onto the closest enrolled person.

## Models Used

| Component | Model | Why |
|-----------|-------|-----|
| Face detection | SCRFD (InsightFace `buffalo_l` pack) | Accurate across face sizes and poses, fast on CPU through ONNX Runtime, and it also returns 5 facial landmarks, which are used to align each face before embedding. |
| Face embedding | ArcFace ResNet-50 (`w600k_r50`, trained on WebFace600K) | ArcFace's angular-margin loss pulls images of the same person together and pushes different people apart, which is exactly what similarity-based identification needs. It gives a 512-d embedding and is pretrained, so no training is required and new people can be enrolled instantly. |
| Similarity metric | Cosine similarity | ArcFace embeddings are optimized for angular separation, and cosine similarity gives a bounded score that is easy to threshold. |
| Runtime | ONNX Runtime | Free, runs on CPU or the free Colab GPU, and needs no API keys. |

## Matching Threshold

- **Metric:** cosine similarity between L2-normalized ArcFace embeddings.
- **Threshold:** **<<PASTE THE THRESHOLD VALUE FROM YOUR CODE>>**
- **Decision rule:** best similarity ≥ threshold → return that identity; otherwise → `unknown`.
- **Trade-off:** a higher threshold reduces false accepts (strangers matched to enrolled people) but increases false rejects (enrolled people marked unknown). A lower threshold does the opposite.

## Setup

```bash
pip install insightface onnxruntime opencv-python-headless "numpy<2" matplotlib scikit-learn Pillow
```

Or open the Colab notebook above and run all cells. No paid services or API keys are used.

## Evaluation Results

<<PASTE THE ACCURACY / FALSE ACCEPT / FALSE REJECT OUTPUT FROM YOUR EVALUATION CELL>>

## Failure Cases

- **Poor lighting or strong shadows** lower embedding quality and similarity scores, which can push a true match below the threshold.
- **Extreme head pose** (profile view, looking far up or down) makes detection and matching less reliable.
- **Occlusion** (masks, sunglasses, hands, hats) hides key facial features.
- **Low resolution or motion blur** produces unstable embeddings.
- **Look-alikes and relatives** can score above the threshold and cause false accepts.
- **Few enrollment images per person** makes matching sensitive to one unusual photo.
- **Missed detections** on very small faces in crowded images.

## Improvements

- Enroll **multiple images per person** and average or match against all their embeddings.
- **Calibrate the threshold** on a larger validation set with more unknown faces, or use per-person adaptive thresholds.
- Add **image quality checks** (blur, brightness, face size) and reject unusable inputs.
- Use **FAISS / approximate nearest neighbours** for fast search when the database grows large.
- Add **liveness / anti-spoofing** to reject photos of photos.
- Wrap the system in a **Gradio or FastAPI** app for easier demos.

## Design Decisions

- **Pretrained embeddings instead of training a classifier:** training from scratch isn't practical in 3 days with no budget, and new people can be enrolled without retraining.
- **Similarity search plus a threshold:** a closed-set classifier would always output some enrolled name, which is wrong for strangers. The threshold gives an explicit "unknown" outcome.
- **Landmark alignment before embedding:** ArcFace expects aligned 112×112 faces, and aligning improves consistency across poses.
- **Free tooling only:** everything runs on open-source libraries on CPU or free Colab GPU.

## Privacy

Face data is sensitive biometric data. This project is for educational use; only use images of people who have consented, and don't commit private photos to the repository.

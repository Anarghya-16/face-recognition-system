# Face Recognition Identification System

A lightweight face identification system that **enrolls** people into a database and **identifies** new faces by matching them against the enrolled identities, with an **"unknown" rejection** mechanism for faces that are not enrolled.

Built for the AI/ML Intern assignment at Code Nimbus Solutions, using only free and open-source tools (₹0 / $0 spend).

- **Colab notebook (full walkthrough + results):** https://colab.research.google.com/drive/1hOi5wZjDx4D0evW1W7VS1qL6WhtEcKKC?usp=sharing
- **Code:** see the [`face_id/`](face_id/) folder

---

## Table of Contents
1. [Overview](#overview)
2. [Pipeline](#pipeline)
3. [Models Used](#models-used)
4. [Matching Threshold](#matching-threshold)
5. [Setup](#setup)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Evaluation Results](#evaluation-results)
9. [Failure Cases](#failure-cases)
10. [Improvements](#improvements)
11. [Design Decisions](#design-decisions)

---

## Overview

The system supports two operations:

| Operation | What it does |
|-----------|--------------|
| **Enroll** | Takes one or more images of a person, detects the face, computes an embedding, and stores it under that person's name. |
| **Identify** | Takes a new image, detects the face(s), computes embeddings, compares them with all enrolled embeddings, and returns the best match **or "unknown"** if the similarity is below the threshold. |

## Pipeline

```
Image → Face Detection → (Crop / Align) → Face Embedding → Similarity Search → Threshold → Identity or "Unknown"
```

1. **Face detection:** locate faces and crop them.
2. **Embedding:** convert each face crop into a fixed-length vector that represents the identity.
3. **Similarity matching:** compare the query embedding to every enrolled embedding using <<cosine similarity / Euclidean distance>>.
4. **Unknown rejection:** if the best similarity is below the threshold, the face is labelled `unknown` instead of being forced onto the closest enrolled person.

## Models Used

| Component | Model / Library | Why |
|-----------|-----------------|-----|
| Face detector | <<e.g. MTCNN / RetinaFace / OpenCV DNN / MediaPipe>> | <<reason, e.g. accurate, free, runs on CPU>> |
| Embedding model | <<e.g. FaceNet InceptionResnetV1 (VGGFace2) / ArcFace / InsightFace / dlib>> | <<reason, e.g. pretrained, 512-d embeddings, strong accuracy>> |
| Similarity metric | <<Cosine similarity / L2 distance>> | <<reason>> |

All models are pretrained and open-source, so no paid APIs or keys are needed.

## Matching Threshold

- **Metric:** <<cosine similarity>>
- **Threshold used:** **<<e.g. 0.xx>>**
- **How it was chosen:** <<e.g. swept thresholds from 0.3 to 0.9 on a validation set containing enrolled and unknown faces, and picked the value that balanced false accepts and false rejects>>
- **Behaviour:**
  - similarity ≥ threshold → return the best-matching identity
  - similarity < threshold → return `unknown`
- **Trade-off:** a higher threshold reduces false accepts (strangers mistaken for enrolled people) but increases false rejects (enrolled people labelled unknown). A lower threshold does the opposite.

## Setup

```bash
git clone https://github.com/Anarghya-16/face-recognition-system.git
cd face-recognition-system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or run everything without installing anything by opening the [Colab notebook](<<PASTE COLAB LINK>>).

## Usage

> Replace the commands below with the exact ones from your code if they differ.

**Enroll a person**
```bash
python face_id/enroll.py --name "Alice" --images path/to/alice_images/
```

**Identify a face**
```bash
python face_id/identify.py --image path/to/test_image.jpg
```

**Example output**
```
Detected 1 face(s)
Face 1: Alice (similarity: 0.82)
```
```
Detected 1 face(s)
Face 1: unknown (best match: Bob, similarity: 0.31 < threshold <<0.xx>>)
```

## Project Structure

```
face-recognition-system/
├── face_id/
│   ├── <<detector / embedding code>>
│   ├── <<enroll script>>
│   ├── <<identify script>>
│   └── <<evaluation script>>
├── requirements.txt
└── README.md
```

## Evaluation Results

**Setup:** <<number of enrolled identities>> identities, <<N>> enrollment images each, <<M>> test images (including <<K>> images of people who were *not* enrolled to test unknown rejection). Dataset: <<source, e.g. LFW subset / own images / Kaggle dataset>>.

| Metric | Value |
|--------|-------|
| Identification accuracy (enrolled people) | <<xx%>> |
| Unknown rejection rate (correctly rejected strangers) | <<xx%>> |
| False accept rate (stranger accepted as enrolled) | <<xx%>> |
| False reject rate (enrolled person labelled unknown) | <<xx%>> |

<<Optionally add a confusion matrix or similarity-score histogram image here, e.g. `![Score distribution](images/scores.png)`>>

## Failure Cases

Observed or expected situations where the system makes mistakes:

- **Poor lighting / strong shadows** lower embedding quality and similarity scores.
- **Extreme pose** (side profile, looking down) makes detection and matching less reliable.
- **Occlusion** (masks, sunglasses, hands, hats) removes key facial features.
- **Low resolution or motion blur** produces unstable embeddings.
- **Look-alikes / similar faces** (e.g. relatives) can exceed the threshold and cause false accepts.
- **Very few enrollment images** per person makes matching sensitive to a single unusual photo.
- **Small faces or crowded scenes:** the detector can miss faces or return multiple faces.
- <<Add any specific failures you actually observed, with example image names>>

## Improvements

- Add **face alignment** (landmark-based) before computing embeddings.
- Enroll **multiple images per person** and average embeddings or match against all of them.
- Use a stronger model (e.g. ArcFace / InsightFace) for better separation between identities.
- **Calibrate the threshold** per model and dataset, or use per-person adaptive thresholds.
- Use **FAISS / approximate nearest neighbours** for fast search with large databases.
- Add **liveness / anti-spoofing** detection to reject photos of photos.
- Add **image quality checks** (blur, brightness, face size) and reject unusable inputs.
- Build a simple **UI or REST API** (e.g. Gradio / FastAPI) for demos.

## Design Decisions

- **Pretrained models:** training from scratch is impractical in 3 days with zero budget, and pretrained embeddings already generalize well to new identities.
- **Embedding + similarity search instead of a classifier:** new people can be enrolled instantly with no retraining.
- **Explicit unknown class through a threshold:** a closed-set classifier would always output some enrolled name, which is wrong for strangers.
- **Free tooling only:** everything runs on CPU or free Colab GPU with open-source libraries.

## Ethics & Privacy

Face data is sensitive biometric data. This project is for educational purposes only; only use images of people who have consented, and do not commit private photos to the repository.

## Author

**Anarghya** · [GitHub](https://github.com/Anarghya-16)

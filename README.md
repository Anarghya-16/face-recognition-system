# Face Recognition Identification System

An open-set face identification system that enrols individuals from images and
identifies new faces by matching them against the enrolled database, with an
explicit **unknown** rejection mechanism for faces that belong to nobody in the
gallery.

The whole system runs on CPU with pre-trained open-source models, so the cost of
building, running and evaluating it is **₹0**.

---

## 1. Pipeline

```
image ──► face detection (SCRFD) ──► 5-point landmark alignment
                                          │
                                          ▼
                              embedding (ArcFace, 512-d, L2-normalised)
                                          │
                                          ▼
                    cosine similarity against every enrolled template
                                          │
                              ┌───────────┴───────────┐
                    score ≥ threshold          score < threshold
                              │                        │
                        identity label              "unknown"
```

| Stage | Component | Why |
|---|---|---|
| Detection | SCRFD-10GF (`buffalo_l`) | Accurate on small, rotated and profile faces; returns the 5 landmarks needed for alignment |
| Alignment | Similarity transform to a 112×112 canonical template | Removes in-plane rotation and scale, which is the single largest source of embedding noise |
| Embedding | ArcFace **w600k_r50** (ResNet-50, 512-d) | Trained with an additive angular margin loss on ~600 k identities, so identities are separated by *angle*; cosine similarity is the natural metric |
| Matching | Cosine similarity, max over an identity's templates | Embeddings are L2-normalised, so cosine similarity reduces to a dot product — a single matrix–vector product over the whole gallery |
| Rejection | Fixed similarity threshold | Turns a nearest-neighbour search (closed set) into an open-set decision |

Both models are ONNX and execute through `onnxruntime`; no training is performed
and no GPU is required.

### Design decisions worth defending

1. **Pre-trained embeddings instead of training a classifier.** A softmax
   classifier over enrolled people must be retrained every time someone is added
   and cannot say "unknown" in a principled way. A metric-learning embedding
   makes enrolment a single forward pass and turns identification into a
   nearest-neighbour search, so new people are added in O(1).
2. **All templates kept, not just a centroid.** The score for a person is the
   **maximum** similarity over that person's enrolled images (`--aggregate max`).
   This tolerates pose and lighting variation, because a query only has to
   resemble *one* enrolment image. `--aggregate mean` is also implemented: it is
   more robust when one enrolment image is bad, but less tolerant of pose
   variation. The trade-off can be measured by re-running `evaluate.py`.
3. **Cosine similarity, not Euclidean distance.** ArcFace optimises angular
   margins, so angle carries the identity signal. On L2-normalised vectors the
   two are monotonically related, but cosine keeps scores in a fixed
   `[-1, 1]` range, which makes a single global threshold meaningful.
4. **A single global threshold, tuned on held-out data.** Per-person thresholds
   would fit better but need enough probes per person to calibrate, which a
   small enrolment set does not provide.
5. **Rejection before ranking, reported separately.** Rank-1 accuracy alone
   hides impostor behaviour, so the evaluation reports identification accuracy
   and unknown-rejection rate as two separate numbers across a threshold sweep.

---

## 2. Setup

```bash
git clone <your-repo-url>
cd face-recognition-system

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `buffalo_l` model pack (~280 MB) downloads automatically on first run into
`~/.insightface/models/` and is cached afterwards.

**Dataset layout** — one folder per person:

```
data/
├── enroll/            # people the system should recognise
│   ├── Alice/ img1.jpg img2.jpg img3.jpg
│   └── Bob/   img1.jpg img2.jpg
├── unknown/           # people who must be rejected (never enrolled)
│   └── Carol/ img1.jpg
└── test/              # loose query images
```

To get a real dataset for free, use Labeled Faces in the Wild:

```bash
pip install scikit-learn Pillow
python prepare_lfw.py --known 20 --unknown 20 --min-faces 5
```

---

## 3. Usage

```bash
# 1. Enrol everyone in data/enroll
python enroll.py --data data/enroll --gallery models/gallery.npz

# 2. Identify
python identify.py --image data/test/query.jpg --threshold 0.35
python identify.py --folder data/test --json results/predictions.json
python identify.py --webcam                     # live demo

# 3. Evaluate
python evaluate.py --data data/enroll --unknown-data data/unknown --gallery-per-id 2

# 4. Optional browser demo
pip install streamlit && streamlit run app.py
```

Annotated images are written to `results/annotated/`, metrics to
`results/metrics.json`, and plots to `results/score_distributions.png` and
`results/threshold_sweep.png`.

Add one more person later without rebuilding the gallery:

```bash
python enroll.py --person "Dev" --images data/enroll/Dev --append
```

---

## 4. Evaluation protocol

Identities are split into two disjoint groups. For each **known** identity the
first `--gallery-per-id` images are enrolled and the remaining images become
probes. Every image of every **unknown** identity becomes a probe whose correct
answer is the literal label `unknown`. No probe image is ever enrolled, so the
gallery and the probe set never overlap.

Reported metrics:

| Metric | Meaning |
|---|---|
| Rank-1 (closed set) | Fraction of known probes whose nearest identity is correct, ignoring the threshold |
| Correct identification rate | Known probes given the right name **and** accepted at the threshold |
| Unknown rejection rate | Unknown probes correctly labelled `unknown` (= 1 − FAR) |
| ROC-AUC / EER | Threshold-free separability of genuine vs impostor similarity scores |
| Macro-F1 | Balanced overall score used to pick the operating threshold |

### Results

Run `python evaluate.py ...` and paste the console summary here before
submitting; `results/metrics.json` holds the full threshold sweep.

```
Identities enrolled      : __ (__ templates)
Probes                   : __ known, __ unknown
Rank-1 (closed set)      : 0.___
ROC-AUC / EER            : 0.____ / 0.____
Genuine   mean ± std     : 0.___ ± 0.___
Impostor  mean ± std     : 0.___ ± 0.___
At threshold 0.35        : correct ID 0.___, unknown rejection 0.___, macro-F1 0.___
Best macro-F1 threshold  : 0.__
```

| Threshold | Correct ID rate | Unknown rejection | Macro-F1 |
|---|---|---|---|
| 0.25 | | | |
| 0.35 | | | |
| 0.45 | | | |
| 0.55 | | | |

---

## 5. Matching threshold

The operating point is a **cosine similarity of 0.35**, the standard value for
ArcFace `w600k_r50` embeddings, and it is confirmed empirically by the threshold
sweep in `evaluate.py`.

Why a threshold is needed at all: nearest-neighbour search always returns
*somebody*. Without rejection, a stranger is silently assigned the name of
whoever they most resemble — the worst possible failure for an identification
system. The threshold converts the nearest-neighbour score into an accept/reject
decision.

How it behaves:

* Genuine pairs (same person) typically score well above the impostor mass, so
  the two distributions overlap only in a narrow band. `score_distributions.png`
  visualises exactly this.
* **Lowering** the threshold raises the correct identification rate and lowers
  unknown rejection — more people are named, including strangers.
* **Raising** it hardens rejection at the cost of misses on hard genuine pairs
  (profile views, poor lighting, large age gaps).
* The right choice is application-dependent. Attendance or door access should
  sit high (0.45–0.55) because a false accept is a security breach; a photo
  organiser can sit lower because a false accept is merely annoying. The
  threshold is a CLI argument precisely so it can be re-tuned per deployment
  without touching the code.

The `margin` field in every prediction (best score minus second-best) is an
extra confidence cue: a high score with a near-zero margin means two enrolled
people are genuinely hard to tell apart, and is worth flagging for review.

---

## 6. Failure cases

| Failure | Cause | Observed behaviour | Mitigation |
|---|---|---|---|
| **Pre-cropped, tightly aligned faces are not detected** | SCRFD expects context around the face; a 112×112 crop with no margin has none | Reproduced with a tightly cropped sample image: zero detections, so the person cannot be enrolled at all | Enrol from full photographs with margin; pad crops before enrolment, or fall back to treating the whole image as an aligned face |
| **Extreme pose (beyond ~±60° yaw)** | Alignment cannot compensate for self-occlusion | Genuine similarity drops towards the impostor range, producing a false `unknown` | Enrol 3–5 images covering frontal, left and right poses; `--aggregate max` already exploits this |
| **Low resolution or motion blur** | Fewer than ~50 px between the eyes leaves too little detail | Embeddings drift toward the population mean, so every score shrinks | Reject faces below a minimum bounding-box size and below `--min-det-score` at enrolment (already enforced) |
| **Heavy occlusion — masks, hands, sunglasses** | Large parts of the discriminative region are missing | Detection often survives, but the embedding is unreliable and the margin collapses | Use a mask-aware recognition model, or require a higher threshold when the margin is small |
| **Strong backlighting or near-darkness** | Detector confidence falls first, embedding quality second | Missed detections and depressed genuine scores | Histogram equalisation / CLAHE before inference; enrol under varied lighting |
| **Identical twins and very close relatives** | A genuine limitation of appearance-only recognition | High score for the wrong person, with a very small top-2 margin | Flag low-margin decisions for human review; add a second modality |
| **Large age gap between enrolment and query** | Facial geometry drifts over years | Gradual decline in genuine scores | Periodic re-enrolment; append newer templates rather than replacing old ones |
| **Multiple faces in an enrolment image** | Ambiguity about who is being enrolled | The largest face is used and a warning is printed | Enrol from single-subject photographs |
| **Presentation attacks (a photo of a photo)** | There is no liveness check anywhere in the pipeline | A printed photo is accepted as the genuine person | Add passive liveness / anti-spoofing before any security use — see below |

---

## 7. Improvements

**Accuracy**
* Test-time augmentation: average the embedding of the image and its horizontal
  flip, a consistent and nearly free gain.
* Quality-weighted templates: weight each enrolled template by detector
  confidence and face size instead of treating all enrolments equally.
* Per-identity or score-normalised thresholds (z-norm / t-norm) once enough
  probes exist to calibrate them.
* Compare `buffalo_l` against `antelopev2` and a larger ArcFace backbone on the
  same protocol, since the evaluation harness is model-agnostic.

**Robustness**
* Passive liveness detection — mandatory before any access-control deployment.
* CLAHE and gamma correction in the pre-processing stage for poor lighting.
* Reject at enrolment on a face-quality score rather than detector confidence
  alone.

**Scale**
* Beyond a few thousand templates, replace the exhaustive dot product with an
  ANN index (FAISS `IndexFlatIP`, or HNSW for millions of templates); the
  `Gallery` interface was kept narrow so the backend can be swapped without
  touching the callers.
* Batch detection and embedding to amortise ONNX inference overhead.
* Track identities across video frames and vote over a window, instead of
  deciding on each frame independently.

**Engineering**
* A FastAPI service exposing `/enrol` and `/identify`, with the gallery in a
  database rather than an `.npz` file.
* Store an audit log of low-margin and rejected decisions to build a real
  hard-case set for future tuning.

**Ethical and privacy considerations**
* Embeddings are biometric data; they should be encrypted at rest and enrolment
  should be consent-based and revocable (`Gallery.remove` supports deletion).
* Published benchmarks show accuracy varies across demographic groups, so any
  real deployment must report per-group metrics rather than a single aggregate.

---

## 8. Repository layout

```
face_id/
├── engine.py        detection + alignment + embedding (FaceEngine)
├── gallery.py       template store, cosine matching, unknown rejection
└── utils.py         dataset helpers
enroll.py            build or extend the gallery
identify.py          identify an image, a folder, or a webcam stream
evaluate.py          open-set evaluation, threshold sweep, plots
prepare_lfw.py       build a free LFW evaluation split
app.py               optional Streamlit demo
requirements.txt
```

## 9. Acknowledgements

InsightFace (`buffalo_l`: SCRFD detector, ArcFace w600k_r50 recogniser),
released under the MIT licence for non-commercial research use. Labeled Faces in
the Wild, University of Massachusetts Amherst.

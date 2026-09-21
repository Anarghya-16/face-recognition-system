"""Optional Streamlit demo: enrol a person and identify an uploaded photo.

    pip install streamlit
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from face_id import FaceEngine, Gallery

GALLERY_PATH = Path("models/gallery.npz")


@st.cache_resource(show_spinner="Loading the face model ...")
def get_engine() -> FaceEngine:
    return FaceEngine()


def load_gallery() -> Gallery:
    return Gallery.load(GALLERY_PATH) if GALLERY_PATH.exists() else Gallery()


def to_bgr(uploaded) -> np.ndarray:
    buf = np.frombuffer(uploaded.read(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


st.set_page_config(page_title="Face Recognition Identification System", layout="centered")
st.title("Face Recognition Identification System")

engine = get_engine()
gallery = load_gallery()
threshold = st.sidebar.slider("Matching threshold (cosine similarity)", 0.10, 0.80, 0.35, 0.01)
aggregate = st.sidebar.selectbox("Template aggregation", ["max", "mean"])
st.sidebar.write(f"Enrolled: **{len(gallery.identities)}** identities, "
                 f"**{len(gallery)}** templates")

tab_enrol, tab_identify = st.tabs(["Enrol", "Identify"])

with tab_enrol:
    name = st.text_input("Person's name")
    files = st.file_uploader("Face images (2-5 recommended)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="enrol")
    if st.button("Enrol", disabled=not (name and files)):
        added = 0
        for f in files:
            face = engine.largest_face(to_bgr(f))
            if face is None:
                st.warning(f"No face detected in {f.name}; skipped.")
                continue
            gallery.add(face.embedding, name, source=f.name)
            added += 1
        if added:
            gallery.save(GALLERY_PATH)
            st.success(f"Enrolled {added} template(s) for {name}.")

with tab_identify:
    query = st.file_uploader("Image to identify", type=["jpg", "jpeg", "png"], key="query")
    if query is not None:
        if len(gallery) == 0:
            st.error("The gallery is empty. Enrol at least one person first.")
        else:
            img = to_bgr(query)
            faces = engine.detect(img)
            if not faces:
                st.warning("No face detected.")
            for face in faces:
                match = gallery.identify(face.embedding, threshold, aggregate)
                engine.annotate(img, face, match.label, match.score, match.known)
                if match.known:
                    st.success(f"Identified as **{match.label}** "
                               f"(similarity {match.score:.3f})")
                else:
                    st.error(f"**unknown** — nearest identity {match.best_label} "
                             f"scored {match.score:.3f}, below {threshold:.2f}")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

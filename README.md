# Face Expression Recognition — AI Portfolio Demo

A Python web app that accepts a user-uploaded face image, detects faces with OpenCV, and classifies **visible facial expressions** using the pretrained DeepFace emotion model. Results show the top predicted expression and a chart of scores for each detected face.

> **Important:** An image cannot establish a person's true emotional state or diagnose a mental-health condition. In particular, this app does **not** detect depression. Predictions can be wrong and can vary with lighting, pose, demographics, and image quality. Do not use for hiring, medical, or other consequential decisions.

## Features
- JPG/PNG image upload
- Face detection and expression classification
- Multiple faces supported
- Score chart for each face
- Simple Streamlit interface
- No database: uploaded photos are processed in memory by this demo and are not intentionally saved by the app

## Tech stack
Python, Streamlit, OpenCV, DeepFace, NumPy, Pillow; DeepFace uses a pretrained deep-learning model.

## Run locally

Python **3.10 or 3.11** is recommended for broad compatibility with ML dependencies.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`). On first analysis, the model may download pretrained weights, so internet access is needed initially. ML installation can take several minutes.

## Example output
See `sample_output.png` for an **illustrative mockup**, not a real model prediction. To capture a genuine output screenshot, run the app with a photo you have permission to use.

## Deploy
Streamlit Community Cloud is a convenient option for this Python app: push `app.py` and `requirements.txt` to a public GitHub repository, then deploy the repo with entrypoint `app.py`. Verify the deployment logs if heavy ML dependencies exceed resource limits. **GitHub Pages alone cannot execute Python/Streamlit code.**

## Suggested portfolio description
**Face Expression Recognition (AI/ML):** Built a Python application that detects faces in uploaded images and classifies visible facial expressions using a pretrained deep-learning model, with interactive score visualizations.

## Privacy
Only upload photos you have permission to process. A public hosted deployment may transmit the image to its hosting provider. The demo does not intentionally write uploads to disk, but hosting infrastructure may have its own logging and retention policies.

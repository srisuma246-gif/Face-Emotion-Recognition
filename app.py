"""Face expression classification demo. Run: streamlit run app.py"""
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Face Expression Recognition', page_icon='😊', layout='wide')
st.title('😊 Face Expression Recognition')
st.caption('Upload a photo to classify visible facial expressions using a pretrained model.')
st.info('Expression labels are estimates from an image, not reliable readings of feelings or mental health. This app does not diagnose depression.')

uploaded = st.file_uploader('Upload a JPG or PNG face photo', type=['jpg', 'jpeg', 'png'])
if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.getvalue())).convert('RGB')
        if image.width * image.height > 20_000_000:
            st.error('Please upload an image smaller than 20 megapixels.')
            st.stop()
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(image, caption='Uploaded photo', use_container_width=True)
        if st.button('Analyze facial expression', type='primary'):
            with st.spinner('Loading model and analyzing... (first run may download weights)'):
                from deepface import DeepFace
                results = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    detector_backend='opencv',
                    enforce_detection=True,
                    align=True,
                    silent=True,
                )
            if not isinstance(results, list):
                results = [results]
            st.success(f'Found {len(results)} face(s).')
            for i, result in enumerate(results, 1):
                st.subheader(f'Face {i}: {result["dominant_emotion"].title()}')
                scores = result.get('emotion', {})
                st.bar_chart({k.title(): float(v) for k, v in scores.items()})
                st.caption('Scores are model estimates, not calibrated probabilities or clinical findings.')
    except ValueError as exc:
        st.warning(f'Could not detect a face clearly. Try a front-facing, well-lit photo. Details: {exc}')
    except Exception as exc:
        st.error(f'Analysis failed: {exc}')
        st.caption('Check dependencies and internet access for first-time model download.')

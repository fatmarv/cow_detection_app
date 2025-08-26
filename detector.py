from ultralytics import YOLO
import streamlit as st
import numpy as np
from PIL import Image

st.title("🐄 Cow Detection App")

# Load model
model = YOLO("best.pt")  #trained model

# --- Session state to remember selection ---
if "input_type" not in st.session_state:
    st.session_state.input_type = None  # nothing selected by default

st.sidebar.title("Choose Input Type")

# Buttons for selection
if st.sidebar.button("Image Upload", use_container_width=True):
    st.session_state.input_type = "Image Upload"

st.sidebar.button("📸 Image Capture (Coming Soon)", disabled=True, use_container_width=True)
st.sidebar.button("🎥 Video Upload (Coming Soon)", disabled=True, use_container_width=True)
st.sidebar.button("🎬 Video Capture (Coming Soon)", disabled=True, use_container_width=True)

# --- Handle feature logic ---
if st.session_state.input_type == "Image Upload":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        # Run detection
        results = model.predict(np.array(image))

        # Get number of cows detected
        cow_count = len(results[0].boxes)

        # Draw detections
        st.image(results[0].plot(), caption="Detection Result")

        # Show cow count
        st.success(f"✅ Cows detected: {cow_count}")
else:
    st.info("👆 Select an option from the sidebar to get started.")

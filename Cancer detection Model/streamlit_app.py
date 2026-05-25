import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import base64

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

def add_background(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        header {{visibility: hidden;}}
        .stApp {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
        }}
        * {{
            color: white !important;
        }}
        .stFileUploader * {{
            color: black !important;
        }}
        .stButton > button, .stButton > button * {{
            color: black !important;
        }}
        </style>
    """, unsafe_allow_html=True)

add_background(r"D:\Graduation Project\Cancer Model\vit_skin_cancer_model\background.jpg")

st.title("🔬 Skin Cancer Detection")
st.markdown("Upload a dermoscopy image to get a diagnosis prediction.")

uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"])

predict_clicked = st.button("🔍 Predict")

    
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing image..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            result = response.json()

        st.success(f"**Prediction:** {result['prediction']}")
        st.info(f"**Confidence:** {result['confidence']}")

        st.subheader("All Probabilities")
        probs = result["all_probabilities"]
        sorted_probs = sorted(probs.items(), key=lambda x: float(x[1].replace("%", "")), reverse=True)

        for label, prob in sorted_probs:
            value = float(prob.replace("%", ""))
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(value / 100, text=label)
            with col2:
                st.markdown(f"**{prob}**")

        if "Malignant" in result["prediction"]:
            st.error("⚠️ Malignant condition detected! Please consult a dermatologist immediately.")
        elif "Pre-malignant" in result["prediction"]:
            st.warning("⚠️ Pre-malignant condition detected! Medical consultation recommended.")
        else:
            st.success("✅ Benign condition detected.")
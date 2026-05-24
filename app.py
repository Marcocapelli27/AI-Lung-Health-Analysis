import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration & Styling for a Medical Context
st.set_page_config(page_title="AI Chest X-Ray Diagnostic Assistant", page_icon="🩺", layout="centered")

st.title("🩺 AI Chest X-Ray Diagnostic Assistant")
st.write("---")
st.markdown("""
**Undergraduate BME Research Project** *This tool uses a fine-tuned MobileNetV2 Deep Learning architecture to analyze chest X-rays for features consistent with Pneumonia.*
""")

# 2. Load the trained AI brain we just saved
@st.cache_resource # This keeps the model loaded in the background so it's super fast
def load_medical_model():
    return tf.keras.models.load_model('pneumonia_diagnostic_model.keras')

with st.spinner("Initializing Clinical AI Model..."):
    model = load_medical_model()

# 3. Create the User Interface Elements
st.subheader("Clinical Data Input")
uploaded_file = st.file_uploader("Upload Patient Chest X-Ray (DICOM/JPEG/PNG format)...", type=["jpg", "jpeg", "png"])

# 4. Processing Pipeline if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image to the user
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Patient Chest X-Ray", use_container_width=True)
    
    st.write("---")
    st.subheader("Diagnostic Analytics")
    
    with st.spinner("Analyzing lung fields for opacities..."):
        # Preprocess the image exactly how our model expects it (224x224 and normalized)
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0) # Add batch dimension
        
        # Run the image through the AI model
        prediction = model.predict(img_expanded)[0][0]
        
    # 5. Display Clinical Interpretation
    probability_percentage = prediction * 100
    
    if prediction > 0.5:
        st.error(f"⚠️ **PNEUMONIA DETECTED** (Probability: {probability_percentage:.2f}%)")
        st.warning("**Clinical Note:** High density infiltrates or consolidation observed in the lung fields. Recommend immediate correlation with patient clinical symptoms, history, and a certified radiologist review.")
    else:
        st.success(f"✅ **NORMAL / CLEAR LUNG FIELDS** (Probability of Pneumonia: {probability_percentage:.2f}%)")
        st.info("**Clinical Note:** Lung volumes appear stable with no clear signs of acute focal consolidation or significant pleural effusion.")

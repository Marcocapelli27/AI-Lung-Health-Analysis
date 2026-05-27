import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import json

# --- Page Setup (Removed the 'X' emoji and kept it clean) ---
st.set_page_config(page_title="Multi-Disease AI Radiology Space", page_icon="🫁", layout="wide") # Changed to wide layout for split-screen

st.title("🫁 Advanced Multi-Disease Chest X-Ray Analyzer")
# Updated description to be highly descriptive but similar in length
st.write("Clinical decision support system providing deep visual feature extraction and differential diagnosis for acute thoracic conditions.")

st.markdown("---")

# --- 1. LOAD MULTI-CLASS MODEL ---
@st.cache_resource
def load_multi_model():
    import os
    import urllib.request
    
    model_filename = "multi_disease_diagnostic_model.keras"
    # Uses your permanent GitHub Releases vault link
    model_url = "https://github.com/Marcocapelli27/AI-Lung-Health-Analysis/releases/download/v2.0/multi_disease_diagnostic_model.keras"
    
    if not os.path.exists(model_filename):
        with st.spinner("Downloading heavy multi-disease matrix architecture from secure release vault..."):
            urllib.request.urlretrieve(model_url, model_filename)
            
    return tf.keras.models.load_model(model_filename)

try:
    model = load_multi_model()
except Exception as e:
    st.error(f"Error loading multi-disease configuration: {e}")

# --- CREATE SPLIT SCREEN COLUMNS ---
# Left Column takes 45% width for the X-ray and stats; Right Column takes 55% for the chat workspace
col1, col2 = st.columns([45, 55], gap="large")

# Global memory string passed down to instruct the live chatbot brain
diagnostic_context = "No image evaluated yet."

# --- LEFT SIDE: IMAGE PREPROCESSING & DIAGNOSTICS ---
with col1:
    st.subheader("📸 Radiograph Analysis Workspace")
    uploaded_file = st.file_uploader("Upload Patient Chest Radiograph...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Case Target", use_container_width=True)
        
        with st.spinner("Executing tensor matrix transformations..."):
            img_resized = image.resize((224, 224))
            img_rgb = img_resized.convert("RGB")
            img_array = np.array(img_rgb) / 255.0
            img_reshaped = np.reshape(img_array, (1, 224, 224, 3))
            
            raw_predictions = model.predict(img_reshaped)[0]
            
        class_labels = ['COVID-19', 'Normal Lung Fields', 'Lung Opacity (Non-COVID Infection)', 'Viral Pneumonia']
        top_prediction_idx = np.argmax(raw_predictions)
        primary_finding = class_labels[top_prediction_idx]
        confidence_score = raw_predictions[top_prediction_idx] * 100
        
        diagnostic_context = f"Primary Finding: {primary_finding} ({confidence_score:.1f}% confidence). " \
                             f"Full Breakdown Profile: " + ", ".join([f"{lbl}: {val*100:.1f}%" for lbl, val in zip(class_labels, raw_predictions)])
        
        st.markdown("### 📊 Automated Neural Network Diagnostics")
        
        # Cleaned up alert banners (Removed heavy colored backgrounds)
        if primary_finding == 'Normal Lung Fields':
            st.info(f"**Clear Screening:** {primary_finding} detected.")
        else:
            st.error(f"**Pathology Alert:** High structural correlation with {primary_finding}.")
            
        # Render neat metric bars with exact percentages right next to the labels
        for lbl, score in zip(class_labels, raw_predictions):
            pct_text = f"{lbl} — **{score*100:.1f}%**"
            st.write(pct_text)
            st.progress(float(score))

# --- RIGHT SIDE: EXPERT CHAT SPACE ---
with col2:
    st.subheader("💬 Expert Consult Chat")
    
    # Persistent chat memory system
    if "messages" not in st.session_state:
        st.session_state.messages = [
            # Swapped out the robot introductory box text for a cleaner presentation
            {"role": "assistant", "content": "Welcome to the clinical workspace. The multi-disease differential matrix has been initialized. Let's discuss structural differences or triage pathways."}
        ]

    # Displays chat history directly inside column 2
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Live Interaction Loop (Gemini Bridge)
    if user_prompt := st.chat_input("Inquire about this complex matrix breakdown..."):
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting Clinical AI Knowledge Base..."):
                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

                    system_instruction = (
                        "You are an expert clinical AI radiologist assistant. "
                        f"Current Case Analytical Context: {diagnostic_context}. "
                        "Instructions: Answer the clinician's or patient's queries thoroughly, with deep biochemical and structural explanations. "
                        "When discussing findings, highlight differences between ground-glass opacities (COVID-19), standard consolidations (Pneumonia), "
                        "and clear fields (Normal). Warn users that neural networks offer probabilities, not legal diagnoses, and must be verified by a physician."
                    )

                    history_context = ""
                    for msg in st.session_state.messages[:-1]:
                        history_context += f"{msg['role'].upper()}: {msg['content']}\n"

                    payload = {
                        "contents": [{
                            "parts": [{
                                "text": f"{system_instruction}\n\nChat History:\n{history_context}\nUser Question: {user_prompt}"
                            }]
                        }]
                    }
                    
                    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
                    ai_reply = response.json()['candidates'][0]['content']['parts'][0]['text']
                    st.markdown(ai_reply)

                except Exception as e:
                    ai_reply = "Connection disruption with the cloud knowledge base matrix."
                    st.error(f"Trace: {e}")
                    st.markdown(ai_reply)
                
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

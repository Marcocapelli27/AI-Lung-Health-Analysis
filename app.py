import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import json

st.set_page_config(page_title="Multi-Disease AI Radiology Space", page_icon="🫁", layout="centered")
st.title("🫁 Advanced Multi-Disease Chest X-Ray Analyzer")
st.write("Inference engine for classifying COVID-19, Viral Pneumonia, Lung Opacities, or Normal structural variants.")

# --- 1. LOAD MULTI-CLASS MODEL ---
@st.cache_resource
def load_multi_model():
    return tf.keras.models.load_model("multi_disease_diagnostic_model.keras")

try:
    model = load_multi_model()
except Exception as e:
    st.error(f"Error loading multi-disease configuration: {e}")

# --- 2. IMAGE PREPROCESSING PIPELINE ---
uploaded_file = st.file_uploader("Upload Patient Chest Radiograph...", type=["jpg", "jpeg", "png"])

# Global memory string passed down to instruct the live chatbot brain
diagnostic_context = "No image evaluated yet."

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Case Target", use_container_width=True)
    
    with st.spinner("Executing tensor matrix transformations..."):
        # Match our 224x224x3 training dimension requirements exactly
        img_resized = image.resize((224, 224))
        img_rgb = img_resized.convert("RGB")
        img_array = np.array(img_rgb) / 255.0
        img_reshaped = np.reshape(img_array, (1, 224, 224, 3))
        
        # Run inference - returns an array of 4 distinct probabilities
        raw_predictions = model.predict(img_reshaped)[0]
        
    # Mapping coordinates to match dataset folders alphabetically
    class_labels = ['COVID-19', 'Normal Lung Fields', 'Lung Opacity (Non-COVID Infection)', 'Viral Pneumonia']
    
    # Identify index with highest probability score
    top_prediction_idx = np.argmax(raw_predictions)
    primary_finding = class_labels[top_prediction_idx]
    confidence_score = raw_predictions[top_prediction_idx] * 100
    
    # Save a clean metric breakdown text for Gemini to absorb
    diagnostic_context = f"Primary Finding: {primary_finding} ({confidence_score:.1f}% confidence). " \
                         f"Full Breakdown Profile: " + ", ".join([f"{lbl}: {val*100:.1f}%" for lbl, val in zip(class_labels, raw_predictions)])
    
    # Render interactive metrics to clinical staff
    st.markdown("### 📊 Automated Neural Network Diagnostics")
    if primary_finding == 'Normal Lung Fields':
        st.success(f"**Clear Screening:** {primary_finding} detected with {confidence_score:.1f}% confidence.")
    else:
        st.error(f"**Pathology Alert:** High structural correlation with {primary_finding} ({confidence_score:.1f}% confidence).")
        
    # Render neat metric bars for all classes
    for lbl, score in zip(class_labels, raw_predictions):
        st.write(f"{lbl}")
        st.progress(float(score))

    st.markdown("---")
    st.subheader("👨‍⚕️ Multi-Disease Expert Chat Space")

    # --- 3. PERSISTENT CHAT MEMORY SYSTEM ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome back to the clinical workspace. I have reviewed the multi-disease differential matrix. Feel free to ask me deep questions about the differences between these findings, structural features, or recommended triage considerations."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- 4. LIVE INTERACTION LOOP (GEMINI BRIDGE) ---
    if user_prompt := st.chat_input("Inquire about this complex matrix breakdown..."):
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Consulting Clinical AI Knowledge Base..."):
                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

                    # Master clinical prompt updated to instruct the model on handling 4 conditions
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

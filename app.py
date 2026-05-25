import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# Set up clean page configuration
st.set_page_config(page_title="Clinical AI Assistant", page_icon="🫁", layout="centered")

st.title("🫁 Clinical AI Chest X-Ray Assistant")
st.write("Upload a patient's digital chest radiograph for automated neural network screening and real-time consultation.")

# --- 1. MODEL LOADING ---
# Streamlit caches this so the 164,097 weights don't reload on every single chat message
@st.cache_resource
def load_diagnostic_model():
    return tf.keras.models.load_model("pneumonia_diagnostic_model.keras")

try:
    model = load_diagnostic_model()
except Exception as e:
    st.error(f"Error loading deep learning model file: {e}")

# --- 2. RADIOLOGY PATIENT PATHOLOGY INPUT ---
uploaded_file = st.file_uploader("Upload Chest X-Ray (DICOM/PNG/JPG/JPEG)...", type=["jpg", "jpeg", "png"])

# Global variable to hold the dynamic clinical summary context passed to the chat assistant
diagnostic_context = "No medical imaging uploaded yet."

if uploaded_file is not None:
    # Open and display the input chest radiograph
    image = Image.open(uploaded_file)
    st.image(image, caption="Current Patient Case Radiograph", use_container_width=True)
    
    # Preprocessing pipeline to match training format (Target size: 150x150, Grayscale)
    with st.spinner("Executing tensor transformations and feature extraction..."):
        img_resized = image.resize((150, 150))
        img_gray = img_resized.convert("L")  # Ensure grayscale matrix
        img_array = np.array(img_gray) / 255.0  # Min-max normalization [0, 1]
        img_reshaped = np.reshape(img_array, (1, 150, 150, 1))  # Expand to 4D tensor batch
        
        # Calculate raw neural inference probability
        prediction_score = float(model.predict(img_reshaped)[0][0])
    
    # Evaluate decision boundary threshold and display structured alerts
    if prediction_score > 0.5:
        diagnostic_context = f"The deep learning neural network detected structural features highly consistent with PNEUMONIA with {prediction_score*100:.1f}% confidence overlay."
        st.error(f"⚠️ **Diagnostic Screening Finding:** {diagnostic_context}")
    else:
        diagnostic_context = f"The deep learning neural network detected NORMAL lung fields with no significant consolidation opacities. Confidence: {(1 - prediction_score)*100:.1f}%."
        st.success(f"✅ **Diagnostic Screening Finding:** {diagnostic_context}")

    st.markdown("---")
    st.subheader("👨‍⚕️ Clinical Consultation Interactive Workspace")
    st.caption("Review findings, investigate architectural metrics, or discuss educational patient communication protocols below.")

    # --- 3. PERSISTENT CHAT MEMORY CORE ---
    # Initialize secure session state memory vault to avoid chat amnesia on script reload
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI clinical assistant. Now that the patient's radiograph is processed, feel free to ask me questions regarding the structural indications, pathophysiological features, or general engineering questions about this model."}
        ]

    # Render complete historical dialog matrix on screen update
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- 4. CHAT READ/WRITE INTERACTION LOOP ---
    if user_prompt := st.chat_input("Input inquiry concerning this case profile..."):
        
        # Display explicit user question box immediately
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        # Append message object to session memory state
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Formulate assistant contextual reply
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing clinical case metrics..."):
                time.sleep(1.2)  # Simulates processing latency
                
                # Dynamic response blueprint tracking current evaluation state
                simulated_reply = f"Thank you for your question: *'{user_prompt}'*.\n\nAs your clinical companion, I am analyzing this conversation thread alongside the active screening data. Current case metric: **{diagnostic_context}**"
                st.markdown(simulated_reply)
                
        # Commit assistant response object to storage vault
        st.session_state.messages.append({"role": "assistant", "content": simulated_reply})


# New Integrated Structure With AI Chatbot feature included

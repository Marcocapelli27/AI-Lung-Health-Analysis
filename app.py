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
        img_resized = image.resize((224, 224))
        img_rgb = img_resized.convert("RGB")  # Ensure 3-channel color matrix to match model
        img_array = np.array(img_rgb) / 255.0  # Min-max normalization [0, 1]
        img_reshaped = np.reshape(img_array, (1, 224, 224, 3))  # Expand to 4D tensor batch matching (None, 224, 224, 3)
        
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
        
        # 1. Display the user's message immediately inside the chat UI
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        # 2. Append user message object to session memory state
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # 3. Formulate live assistant contextual reply using Gemini API
        with st.chat_message("assistant"):
            with st.spinner("Consulting Clinical AI Knowledge Base..."):
                try:
                    import requests
                    import json

                    # Fetch your secure API key from the Streamlit Secrets vault
                    api_key = st.secrets["GEMINI_API_KEY"]
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

                    # This is the master prompt that builds the medical persona and provides the image context
                    system_instruction = (
                        "You are an expert clinical AI assistant specialized in chest radiology and biomedical engineering. "
                        f"Context: A patient's chest X-ray was uploaded and processed by your custom CNN model. Finding: {diagnostic_context}. "
                        "Instructions: Answer the user's questions deeply, accurately, and scientifically using this case context. "
                        "If asked about specific indications, explain what patches or opacities are typically found in pneumonia vs other conditions. "
                        "Maintain strict safety: Never prescribe medication or give a formal diagnosis. Always include an educational caveat "
                        "that findings must be verified by a board-certified radiologist."
                    )

                    # Gather the conversation history so the model remembers past questions
                    history_context = ""
                    for msg in st.session_state.messages[:-1]:
                        history_context += f"{msg['role'].upper()}: {msg['content']}\n"

                    # Package the data to send to Google
                    payload = {
                        "contents": [{
                            "parts": [{
                                "text": f"{system_instruction}\n\nChat History:\n{history_context}\nPatient/Doctor Question: {user_prompt}"
                            }]
                        }]
                    }
                    headers = {'Content-Type': 'application/json'}

                    # Post request to Gemini
                    response = requests.post(url, headers=headers, data=json.dumps(payload))
                    response_json = response.json()

                    # Extract the text answer from the JSON response object
                    ai_reply = response_json['candidates'][0]['content']['parts'][0]['text']
                    st.markdown(ai_reply)

                except Exception as e:
                    ai_reply = "I apologize, but I encountered a connection error retrieving the clinical data matrix. Please check your API key configuration."
                    st.error(f"System Error Trace: {e}")
                    st.markdown(ai_reply)
                
        # 4. Commit the real AI response object to the storage vault
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

# New Integrated Structure With AI Chatbot feature included

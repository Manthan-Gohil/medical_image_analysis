import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
from model import build_model
import google.generativeai as genai


st.set_page_config(
    page_title="AI Radiology Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
# gemini api key
GOOGLE_API_KEY = "AIzaSyBlF0kJqS-OBRDMtDhrXbrIQXXNDaCr_zE" 
MODEL_PATH = "saved_models/fracture_detection_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['Normal', 'Fracture']

# Configure Gemini API if the key is provided
if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.sidebar.error(f"Error configuring Google API: {e}")


# model and helper function
@st.cache_resource
def load_classifier():
    """Loads the trained PyTorch classification model."""
    model = build_model(pretrained=False)
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"Model file not found at {MODEL_PATH}. Please ensure train.py ran successfully.")
        return None
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_image(image, classifier_model):
    """Runs the image through the classifier to get a finding and confidence."""
    image = image.convert("RGB")
    image_tensor = inference_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = classifier_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    return CLASS_NAMES[predicted_idx.item()], confidence.item()

def generate_report_with_llm(finding, confidence):
    """Uses the classifier's output to generate a detailed report with Gemini."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
        return "⚠️ **LLM Reporting Disabled:** Please add your Google API Key in the `app.py` file to enable this feature."
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    You are a world-class radiologist and medical expert. An AI vision model has analyzed a musculoskeletal radiograph and provided the following finding.
    
    **AI Finding:** `{finding}`
    **AI Model Confidence:** `{confidence*100:.2f}%`

    Based *only* on this finding, generate a comprehensive and structured radiology report. The report must be clear, professional, and include these five sections in markdown format:

    1.  **### Findings:**
        * Start with a clear statement confirming the AI's primary finding.
        * Elaborate on what this finding typically implies for the given context (musculoskeletal radiograph).

    2.  **### Impression:**
        * Provide a concise primary diagnosis based on the finding.
        * List 1-2 potential differential diagnoses if applicable (e.g., for 'Fracture', consider types like hairline, transverse, etc.).

    3.  **### Recommendations:**
        * Suggest immediate next steps for the patient or referring physician.
        * Example: "Urgent clinical correlation is recommended." or "Consultation with an orthopedic specialist is advised."

    4.  **### Suggested Prescription/Treatment Plan:**
        * Based on the finding, outline a standard, non-pharmacological treatment plan.
        * For a 'Fracture' finding, suggest: Immobilization (e.g., splint, cast), Pain Management (e.g., RICE protocol - Rest, Ice, Compression, Elevation), and Follow-up imaging schedule.
        * For a 'Normal' finding, suggest: Conservative management, monitoring of symptoms, and follow-up if pain persists.

    5.  **### Disclaimer:**
        * Include this exact text: "This report was generated with the assistance of an AI model and is for informational and educational purposes only. It is not a substitute for a professional medical diagnosis, and all findings must be verified by a qualified human radiologist."
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ **LLM Error:** Could not generate report. Details: {e}"

# =============================================================================
# 4. STREAMLIT UI LAYOUT
# =============================================================================

st.title("AI-Assisted Radiology Report Generator")
st.markdown("Upload a musculoskeletal X-ray to receive an AI-powered analysis and a detailed report.")

st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload your X-ray image", type=["png", "jpg", "jpeg"])

# Load the classifier model once
classifier = load_classifier()

if uploaded_file is not None:
    # Create a two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Uploaded Radiograph")
        image = Image.open(uploaded_file)
        st.image(image, caption="Patient X-ray", use_column_width=True)

    with col2:
        st.header("Analysis and Report")
        if classifier:
            if st.button("✨ Analyze and Generate Report"):
                # Run classifier
                with st.spinner("Running classification model..."):
                    finding, confidence = classify_image(image, classifier)
                
                # Display classifier result
                st.subheader("🤖 AI Classifier Result")
                if finding == "Fracture":
                    st.error(f"**Finding:** {finding}")
                else:
                    st.success(f"**Finding:** {finding}")

                st.write("**Confidence Score:**")
                st.progress(confidence, text=f"{confidence*100:.2f}%")

                # Generate and display LLM report
                with st.spinner("Generating detailed report with Gemini..."):
                    report = generate_report_with_llm(finding, confidence)
                
                st.header("📋 Analysis Report", divider="rainbow")
                st.markdown(report)
        else:
            # This message shows if the model .pth file is missing
            st.warning("Classifier model could not be loaded. Please ensure the model has been trained.")
else:
    st.info("Please upload an image using the sidebar to get started.")
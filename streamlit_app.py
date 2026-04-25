import os
import warnings
import streamlit as st
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import base64
from huggingface_hub import hf_hub_download
from utils.config import REPO_ID, MODEL_FILENAME
from utils.processing import preprocess_image
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.report_generator import generate_docx_report
from utils.model_architecture import Avg2MaxPooling, DepthwiseSeparableConv
from utils.logger import logger

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="ThyroidAI - Cancer Detection System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Medical Theme ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Outfit:wght@300;500;700&display=swap');

        :root {
            --primary: #007BFF;
            --secondary: #6C757D;
            --success: #28A745;
            --danger: #DC3545;
            --background: #F8F9FA;
            --text: #212529;
        }

        .main {
            background-color: var(--background);
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            color: #1A1E23;
        }

        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3.5em;
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #0056b3;
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
        }

        .prediction-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            border-left: 5px solid var(--primary);
            margin-bottom: 2rem;
        }

        .malignant-card { border-left: 10px solid var(--danger); background-color: #FFF5F5; }
        .benign-card { border-left: 10px solid var(--success); background-color: #F6FFF8; }

        .metric-label { font-size: 0.9rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 1.8rem; font-weight: 800; color: #1A1E23; }

        .sidebar-content {
            padding: 1rem;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            font-size: 0.8rem;
            border-top: 1px solid #dee2e6;
            margin-top: 4rem;
        }

        /* Gradient Text */
        .gradient-text {
            background: linear-gradient(90deg, #007BFF, #00C6FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

@st.cache_resource
def load_model():
    """Loads model from Hugging Face with caching."""
    try:
        logger.info("Loading model for Streamlit...")
        path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        custom_objects = {"Avg2MaxPooling": Avg2MaxPooling, "DepthwiseSeparableConv": DepthwiseSeparableConv}
        model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
        logger.info("Streamlit model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Streamlit model failed to load: {e}")
        return None

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- UI Components ---
def sidebar_nav():
    with st.sidebar:
        st.image("images/banner.png", use_container_width=True)
        st.markdown("### 🧬 ThyroidAI System")
        st.markdown("---")
        selection = st.radio(
            "Navigation",
            ["🏠 Analysis Dashboard", "🔬 Research Methodology", "🧠 Understanding XAI", "⚖️ Disclaimer & Usage"]
        )
        st.markdown("---")
        st.info("Version: 2.0.4-Research\n\nDeveloped for Thyroid Cancer Analysis using Explainable AI.")
        return selection

def dashboard_page(model):
    st.markdown('<h1 class="gradient-text">Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.write("Upload a thyroid ultrasound or pathology image for deep-learning powered malignancy assessment.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Drop thyroid image here (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Specimen", use_container_width=True)

    if uploaded_file:
        with col2:
            st.markdown("### ⚙️ System Processing")
            with st.status("Analyzing Medical Data...", expanded=True) as status:
                st.write("Preprocessing image...")
                processed_img = preprocess_image(image)
                
                st.write("Executing FibonacciNet inference...")
                preds = model.predict(processed_img)
                score = float(preds[0][0])
                is_cancer = score > 0.5
                
                label = "Malignant" if is_cancer else "Benign"
                conf_percent = score * 100 if is_cancer else (1 - score) * 100
                
                st.write("Generating Explainability maps...")
                gradcam_img = None
                try:
                    last_conv = next((l.name for l in model.layers[::-1] if "depthwise_separable_conv" in l.name), None)
                    if last_conv:
                        heatmap = make_gradcam_heatmap(processed_img, model, last_conv)
                        if heatmap is not None:
                            gradcam_img = save_and_display_gradcam(image, heatmap)
                except Exception as e:
                    st.error(f"XAI Error: {e}")
                
                status.update(label="Analysis Complete", state="complete", expanded=False)

            # Results Card
            card_style = "malignant-card" if is_cancer else "benign-card"
            st.markdown(f"""
                <div class="prediction-card {card_style}">
                    <div class="metric-label">Diagnosis Result</div>
                    <div class="metric-value">{label} ({'Positive' if is_cancer else 'Negative'})</div>
                    <div style="margin-top: 10px;">
                        <span class="metric-label">Confidence Score:</span>
                        <span style="font-weight: 600;">{score:.4f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.metric("Confidence Percentage", f"{conf_percent:.2f}%", delta=f"{'HIGH' if conf_percent > 85 else 'MODERATE'}")

        # Explainability Section
        st.markdown("---")
        st.markdown("### 🧠 Interpretability Analysis (Grad-CAM)")
        if gradcam_img:
            gc1, gc2 = st.columns(2)
            gc1.image(image, caption="Original Medical Image", use_container_width=True)
            gc2.image(gradcam_img, caption="AI Focus Area (Heatmap)", use_container_width=True)
            st.caption("The heatmap highlights the regions in the image that most influenced the AI's classification decision.")
        
        # Report Section
        st.markdown("---")
        st.markdown("### 📄 Clinical Report")
        if gradcam_img:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            grad_bytes = io.BytesIO()
            gradcam_img.save(grad_bytes, format='PNG')
            grad_bytes.seek(0)
            
            report = generate_docx_report(img_bytes, f"Predicted: {label}", score, conf_percent, grad_bytes)
            
            st.download_button(
                label="📥 Download Clinical Analysis Report (DOCX)",
                data=report,
                file_name=f"thyroid_report_{label.lower()}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

def research_page():
    st.title("🔬 Research Methodology")
    st.markdown("""
    ### FibonacciNet Architecture
    This system utilizes a custom-designed Deep Convolutional Neural Network (DCNN) dubbed **FibonacciNet**. 
    The filter sizes follow the Fibonacci sequence to efficiently capture features at varying scales.

    #### Key Innovations:
    1. **Avg2Max Pooling Layer**: A novel pooling strategy that emphasizes edge detection by calculating the difference between Average and Max pooling.
    2. **Depthwise Separable Convolutions**: Reduces computational overhead while maintaining high feature extraction performance.
    3. **Fibonacci Scaling**: The network expands its capacity following the sequence: 21, 34, 55, 89, 144, 233, 377.

    #### Dataset Info:
    The model was trained on a curated dataset of thyroid ultrasound and pathology images, with rigorous cross-validation to ensure clinical relevance.
    """)
    st.info("""
    **Technical Insight:** FibonacciNet scaling allows the model to progressively 
    expand its receptive field, mimicking how a radiologist might zoom in from 
    a global view to a specific focal point.
    """)

def xai_page():
    st.title("🧠 Understanding Explainable AI (XAI)")
    st.markdown("""
    ### Why Explainability Matters in Medicine
    In clinical settings, "Black Box" models are difficult to trust. Our system integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual evidence for its decisions.

    #### How to interpret the results:
    - **Red/Yellow Areas**: High influence on the prediction. This is where the AI "sees" malignant or benign characteristics.
    - **Blue/Green Areas**: Low influence on the decision.
    
    If the AI flags an area as **Malignant** and the heatmap overlaps with a visible nodule, it provides the radiologist with a "second opinion" that can be visually verified.
    """)

def disclaimer_page():
    st.title("⚖️ Disclaimer & Usage")
    st.warning("""
    **IMPORTANT CLINICAL NOTICE:**
    This tool is intended for research and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)
    st.markdown("""
    ### Usage Instructions:
    1. **Upload**: Provide a clear ultrasound or pathology image of the thyroid.
    2. **Review**: Check the prediction score and the confidence percentage.
    3. **Verify**: Use the Grad-CAM heatmap to ensure the AI is focusing on the correct clinical features.
    4. **Report**: Download the DOCX report for your records or to share with a clinical peer.
    """)

# --- Main Application Logic ---
def main():
    model = load_model()
    if not model:
        st.error("Model failed to load. Please check your internet connection or repository configuration.")
        st.stop()

    page = sidebar_nav()

    if page == "🏠 Analysis Dashboard":
        dashboard_page(model)
    elif page == "🔬 Research Methodology":
        research_page()
    elif page == "🧠 Understanding XAI":
        xai_page()
    elif page == "⚖️ Disclaimer & Usage":
        disclaimer_page()

    # Footer
    st.markdown("""
        <div class="footer">
            <p>© 2026 ThyroidAI Research Project | Advanced Cancer Detection with Explainable AI</p>
            <p>Built with Streamlit, TensorFlow, and ❤️ for Medical Science</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

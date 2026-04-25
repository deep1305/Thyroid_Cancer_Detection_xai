# 🧬 Research-Based Thyroid Cancer Detection with Explainable AI

An AI-powered thyroid cancer detection system for research and educational use.  
The project supports image-based malignancy prediction with **Explainable AI (Grad-CAM)**, downloadable clinical-style reports, and both **FastAPI** and **Streamlit** interfaces.

## ✨ Features

### 🎯 Core Functionality
- **Thyroid Image Classification**: Predicts malignant vs benign from uploaded thyroid images.
- **Dual Interface Support**: Includes a FastAPI backend + HTML frontend and a rich Streamlit app.
- **Clinical-Style Reporting**: Generates downloadable DOCX reports with prediction details.
- **Production-Friendly Logging**: Centralized logger writes to console and `logs/app.log`.

### 🧠 AI / Research Pipeline
- **Custom Model Architecture**: FibonacciNet-inspired design with custom layers.
- **Custom Layers**:
  - `Avg2MaxPooling`
  - `DepthwiseSeparableConv`
- **Explainable AI**: Grad-CAM heatmaps highlight model focus regions.
- **Image Preprocessing**: Standardized resize (`224x224`) and normalization pipeline.

---

## 🏗️ Project Structure

```text
Research Based Cancer Detection with Explainable AI/
├── app.py                          # FastAPI entrypoint
├── streamlit_app.py                # Streamlit research dashboard
├── requirements.txt
├── pyproject.toml
├── backend/
│   └── routes.py                   # API routes: analyze + report
├── frontend/
│   ├── templates/
│   │   └── index.html              # Web UI template
│   └── static/
│       ├── style.css
│       └── app.js
├── utils/
│   ├── config.py                   # Hugging Face model source config
│   ├── processing.py               # Image preprocessing
│   ├── gradcam.py                  # Explainability heatmap logic
│   ├── model_architecture.py       # Custom model layers/architecture
│   ├── report_generator.py         # DOCX clinical report generator
│   └── logger.py                   # Logging utilities
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+ (as defined in `pyproject.toml`)
- `pip` or `uv`
- Internet connection (required first time to download model from Hugging Face)

### 1) Clone the repository

```bash
git clone https://github.com/deep1305/Thyroid_Cancer_Detection_xai.git
cd "Research Based Cancer Detection with Explainable AI"
```

### 2) Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Using uv (optional):

```bash
uv sync
```

### 3) Run FastAPI web app

```bash
python app.py
```

Open: [http://localhost:8000](http://localhost:8000)

### 4) Run Streamlit app (alternative UI)

```bash
streamlit run streamlit_app.py
```

---

## 🔧 Configuration Notes

Current model source config (`utils/config.py`):

- `REPO_ID = "BWayne1305/Thyroid_Cancer_Detection"`
- `MODEL_FILENAME = "thyroid_cancer_model.keras"`

Inference and preprocessing defaults:

- Input shape: `224x224x3`
- Prediction threshold: `0.5` for malignant/benign split
- Grad-CAM layer selection: last layer containing `depthwise_separable_conv`

---


## 🧪 Example Usage

- Upload thyroid ultrasound or pathology image
- Review:
  - Predicted label (Malignant / Benign)
  - Confidence score and percentage
  - Grad-CAM focus heatmap
- Download generated DOCX report for review/documentation

---

## ⚠️ Clinical Disclaimer

This project is for **research and educational purposes only**.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## 🙌 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face Hub](https://huggingface.co/)
- Grad-CAM explainability community research

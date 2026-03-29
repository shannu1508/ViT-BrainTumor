# 🧠 NeuroVision AI — Brain Tumor Detection with Vision Transformer (ViT)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep-learning powered medical diagnostic system that classifies brain tumors from MRI scans using a custom **Vision Transformer (ViT)** model, visualizes predictions with **Grad-CAM attention heatmaps**, and integrates a local **Ollama AI medical chatbot** — all served through a sleek **Streamlit** web app.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **ViT Classification** | Custom Vision Transformer trained on 4 tumor classes |
| 🔥 **Grad-CAM Heatmaps** | Attention rollout visualization overlay on MRI scans |
| 🤖 **AI Medical Chatbot** | Local Ollama LLM (phi/mistral) for Q&A on results |
| 📚 **Education Library** | Built-in reference cards for each tumor type |
| ⚡ **GPU / CPU Support** | Automatically uses CUDA if available |

---

## 🗂️ Project Structure

```
vit-brain-tumor/
├── streamlit_app.py     # Main Streamlit web application
├── transformer.py       # ViT model definition (TumorClassifierViT)
├── vit_gradcam.py       # Grad-CAM attention heatmap generator
├── train.py             # Model training script
├── test.py              # Standalone inference / evaluation script
├── test_ollama.py       # Ollama connectivity test utility
├── best_model.pth       # Pre-trained model weights (excluded from git)
├── run_app.bat          # One-click launcher (Windows)
├── run_app.sh           # One-click launcher (Linux/macOS)
├── requirements.txt     # Python dependencies
└── data/
    ├── Training/        # Training images (per-class subfolders)
    └── Testing/         # Testing images (per-class subfolders)
```

### Dataset Folder Layout

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

> **Dataset**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — download and extract into `./data/`.

---

## 🚀 Quick Start

### Prerequisites

- Python **3.9 or higher**
- pip
- *(Optional)* NVIDIA GPU + CUDA for faster training/inference
- *(Optional)* [Ollama](https://ollama.com/) for the AI chatbot feature

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/shannu1508/ViT-BrainTumor.git
cd ViT-BrainTumor
```

---

### Step 2 — Create & Activate a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Download the Dataset

1. Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.
2. Extract and place it so the directory structure matches:
   ```
   data/Training/<class>/...
   data/Testing/<class>/...
   ```

---

### Step 5 — Train the Model *(skip if using pre-trained weights)*

```bash
python train.py
```

- Trains for **100 epochs** by default (configurable inside `train.py`).
- Saves the best model to `best_model.pth` and checkpoints to `checkpoint.pth`.
- Generates `loss.png`, `accuracy.png`, and `confusion_matrix.png` plots after training.
- Training automatically **resumes** from the last checkpoint if one exists.

---

### Step 6 — Run the Streamlit App

**Option A — Use the launcher script:**

Windows:
```bash
run_app.bat
```

Linux / macOS:
```bash
bash run_app.sh
```

**Option B — Run directly:**
```bash
streamlit run streamlit_app.py
```

Then open your browser at **http://localhost:8501**

---

### Step 7 — *(Optional)* Enable the AI Chatbot via Ollama

The AI Medical Assistant tab requires [Ollama](https://ollama.com/) running locally.

1. **Install Ollama** from https://ollama.com/download
2. **Start the Ollama server:**
   ```bash
   ollama serve
   ```
3. **Pull a model** (phi is lightweight and recommended):
   ```bash
   ollama pull phi
   ```
4. Refresh the app — the chatbot status in the sidebar will show **Online**.

---

## 🖥️ App Pages

### 1. MRI Analysis
- Upload a brain MRI scan (`.jpg`, `.png`, `.jpeg`, `.tif`).
- The ViT model classifies it into one of four categories.
- Results show confidence score, probability bars, a **Grad-CAM attention heatmap**, and clinical recommendations.

### 2. AI Medical Assistant
- Chat-style interface powered by a local Ollama LLM.
- Automatically receives context from the last MRI analysis result.
- Asks questions like *"What is glioma?"* or *"What are treatment options?"*.

### 3. Education Library
- Reference cards for **Glioma**, **Meningioma**, and **Pituitary** tumor types.

---

## 🧪 Standalone Inference (No Streamlit)

To classify a single image from the command line:

1. Edit `test.py` and set `image_path` to your MRI image path.
2. Run:
   ```bash
   python test.py
   ```
3. This prints the predicted class and confidence, and saves a `prediction_result.png` visualization.

---

## 🏗️ Model Architecture

The model is a **Vision Transformer (ViT)** built with the `vit-pytorch` library:

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 |
| Patch size | 32 × 32 |
| Number of classes | 4 |
| Embedding dimension | 1024 |
| Transformer depth | 6 layers |
| Attention heads | 16 |
| MLP dimension | 2048 |
| Dropout | 0.1 |

**Classes:** `Glioma` · `Meningioma` · `No Tumor` · `Pituitary`

---

## 📊 Training Details

| Setting | Value |
|---|---|
| Optimizer | SGD (lr=0.01) |
| Loss Function | Cross-Entropy |
| Batch size | 32 |
| Epochs | 100 |
| Augmentation | Horizontal flip, ±10° rotation |
| Normalization | ImageNet statistics |

---

## ⚠️ Disclaimer

> This tool is intended for **research and educational purposes only**. It is **not a certified medical device**. All results must be reviewed and verified by a qualified medical professional before any clinical decision-making.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [vit-pytorch](https://github.com/lucidrains/vit-pytorch) — ViT implementation
- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Kaggle
- [Streamlit](https://streamlit.io/) — Web app framework
- [Ollama](https://ollama.com/) — Local LLM inference

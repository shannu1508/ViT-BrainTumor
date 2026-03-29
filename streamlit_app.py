import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
from transformer import TumorClassifierViT
from vit_gradcam import GradCAM

# ==========================================
# 🎨 PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Modern Light Theme
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1F2937;
        background-color: #F7F8FC;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
        border-right: 1px solid #F3F4F6;
    }
    
    /* Cards & Containers */
    .stApp {
        background-color: #F7F8FC;
    }
    
    div.stButton > button {
        background-color: #6C63FF;
        color: white;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(108, 99, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: #5a52d5;
        box-shadow: 0 6px 12px rgba(108, 99, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Upload Area */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border: 2px dashed #E5E7EB;
        text-align: center;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #111827;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    h1 { font-size: 2.5rem; }
    h2 { font-size: 1.5rem; }
    
    /* Metrics & Info Cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #F3F4F6;
    }
    
    /* Custom Classes */
    .premium-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #F3F4F6;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-active {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    
    .progress-bar-container {
        height: 8px;
        background-color: #F3F4F6;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .progress-bar-fill {
        height: 100%;
        background-color: #6C63FF;
        border-radius: 4px;
    }
    
    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 MODEL & BACKEND LOGIC
# ==========================================

# Constants
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Load the Vision Transformer model and weights."""
    model = TumorClassifierViT(num_classes=4)
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model, True
    except Exception as e:
        return None, str(e)

def process_image(image):
    """Preprocess image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def get_heatmap(model, img_tensor):
    """Generate attention heatmap."""
    # This relies on the hook method in your transformer.py
    # Re-implementing simplified hook capture here if needed or using model's method
    if hasattr(model, 'get_last_selfattention'):
        return model.get_last_selfattention(img_tensor)
    return None

@st.cache_data(ttl=300)
def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            return models
        return []
    except:
        return []

def query_ollama_stream(prompt, model="phi"):
    """Query local Ollama instance with streaming."""
    url = "http://localhost:11434/api/generate"
    
    # Auto-select model if specific one not found
    available_models = get_ollama_models()
    if not available_models:
        yield "⚠️ Error: No Ollama models found. Please run `ollama pull phi` or `ollama pull mistral`."
        return
    
    # Use requested model if available, else first available
    if model not in available_models:
        preferred = ['phi', 'mistral', 'llama3', 'gemma']
        selected_model = next((m for m in available_models if any(p in m for p in preferred)), available_models[0])
    else:
        selected_model = model

    payload = {
        "model": selected_model,
        "prompt": prompt,
        "stream": True  # Enable streaming
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        body = json.loads(line)
                        if "response" in body:
                            yield body["response"]
                        if body.get("done", False):
                            break
            else:
                yield f"Error: Ollama returned status {response.status_code}. {response.text}"
    except Exception as e:
        yield f"Error connecting to Ollama: {str(e)}"

def check_ollama_status():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/", timeout=1)
        return response.status_code == 200
    except:
        return False

# Load Model
model, model_status = load_model()

# ==========================================
# 🖥️ UI COMPONENT: SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## 🧠 **NeuroVision AI**")
    st.markdown("<p style='color:#6B7280; margin-top:-15px; font-size:0.9rem;'>Medical Diagnostic System</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "Navigation", 
        ["MRI Analysis", "AI Medical Assistant", "Education Library"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System Status Card
    st.markdown("""
    <div style='background-color: #F9FAFB; padding: 1rem; border-radius: 12px; border: 1px solid #E5E7EB;'>
        <h4 style='margin:0; font-size:0.9rem; color:#374151;'>System Status</h4>
        <div style='display:flex; justify-content:space-between; margin-top:0.5rem; align-items:center;'>
            <span style='font-size:0.8rem; color:#6B7280;'>Model Engine</span>
            <span class='status-badge status-active'>Active</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-top:0.5rem; align-items:center;'>
            <span style='font-size:0.8rem; color:#6B7280;'>Device</span>
            <span style='font-size:0.8rem; font-weight:600; color:#4B5563;'>""" + str(DEVICE).upper() + """</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-top:0.5rem; align-items:center;'>
            <span style='font-size:0.8rem; color:#6B7280;'>Ollama AI</span>
            <span style='font-size:0.8rem; font-weight:600; color:#""" + ("059669" if check_ollama_status() else "DC2626") + """;'>""" + ("Online (" + (get_ollama_models()[0].split(':')[0] if get_ollama_models() else "No Model") + ")" if check_ollama_status() else "Offline") + """</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:auto; padding-top:2rem; font-size:0.75rem; color:#9CA3AF; text-align:center;'>v2.0.0 Premium Build</div>", unsafe_allow_html=True)

# ==========================================
# 🏠 MAIN PAGE: MRI ANALYSIS
# ==========================================

if menu == "MRI Analysis":
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1>AI Brain Tumor Detection</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1rem; color:#6B7280;'>Vision Transformer powered MRI analysis & classification</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='text-align:right; padding-top:1rem;'>
            <span class='status-badge status-active'>System Operational</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Upload Section
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg", "tif"])
    
    if uploaded_file is not None:
        if model is None:
            st.error(f"Failed to load model: {model_status}")
        else:
            # Process Image
            image = Image.open(uploaded_file).convert("RGB")
            img_tensor = process_image(image)
            
            # Prediction
            with st.spinner("Analyzing MRI Pattern..."):
                time.sleep(0.8) # UX smoothing
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)
                    
            pred_class = CLASSES[pred_idx.item()]
            conf_score = conf.item()
            
            # Store prediction in session state for Chatbot context
            st.session_state['last_results'] = {
                "class": pred_class,
                "confidence": conf_score,
                "probabilities": {CLASSES[i]: probs[i].item() for i in range(len(CLASSES))}
            }
            
            # Results Layout
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            col_img, col_res = st.columns([1, 1], gap="large")
            
            with col_img:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.image(image, caption="Original MRI Scan", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_res:
                st.markdown(f"""
                <div class="premium-card">
                    <h3 style='margin-top:0;'>Diagnostic Result</h3>
                    <div style='display:flex; align-items:baseline; gap:10px; margin: 15px 0;'>
                        <span style='font-size:2.5rem; font-weight:700; color:#6C63FF;'>{conf_score*100:.1f}%</span>
                        <span style='font-size:1.2rem; font-weight:500; color:#4B5563;'>Confidence</span>
                    </div>
                    <div style='margin-bottom:20px;'>
                        <span style='background-color:#EEF2FF; color:#6C63FF; padding:5px 15px; border-radius:20px; font-weight:600; font-size:1rem; border:1px solid #C7D2FE;'>
                            Detected: {pred_class}
                        </span>
                    </div>
                    <p style='color:#6B7280; font-size:0.9rem;'>
                        The model has identified patterns consistent with <b>{pred_class}</b>. 
                        Please review the probability distribution below.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability Bars
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown("<h4 style='font-size:1rem; margin-bottom:15px;'>Class Probabilities</h4>", unsafe_allow_html=True)
                
                for i, class_name in enumerate(CLASSES):
                    prob = probs[i].item()
                    color = "#6C63FF" if i == pred_idx else "#E5E7EB"
                    text_color = "#1F2937" if i == pred_idx else "#9CA3AF"
                    font_weight = "600" if i == pred_idx else "400"
                    
                    st.markdown(f"""
                    <div style='margin-bottom:12px;'>
                        <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                            <span style='font-size:0.9rem; color:{text_color}; font-weight:{font_weight}'>{class_name}</span>
                            <span style='font-size:0.9rem; color:{text_color}; font-weight:{font_weight}'>{prob*100:.1f}%</span>
                        </div>
                        <div style='height:8px; width:100%; background-color:#F3F4F6; border-radius:4px;'>
                            <div style='height:100%; width:{prob*100}%; background-color:{color}; border-radius:4px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Heatmap & Recommendations
            st.markdown("### 🔍 Detailed Analysis")
            col_heat, col_rec = st.columns([1, 1], gap="large")
            
            with col_heat:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown("#### Attention Heatmap")
                st.info("Attention map generation feature is active.")
                # Generate and display heatmap overlay
                try:
                    with st.spinner("Generating attention heatmap..."):
                        # Instantiate GradCAM with the loaded model
                        gradcam = GradCAM(model, device=DEVICE)
                        # Prepare image for GradCAM and compute heatmap
                        input_tensor_gc = gradcam._prep_image(image)
                        heatmap = gradcam.generate_heatmap(input_tensor_gc)
                        overlay = GradCAM.overlay_heatmap_on_image(image, heatmap, alpha=0.5)

                    st.image(overlay, caption="Attention Heatmap Overlay", width=400)
                except Exception as e:
                    st.error(f"Failed to generate heatmap: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_rec:
                st.markdown(f"""
                <div class="premium-card" style='border-left: 4px solid {"#10B981" if pred_class == "No Tumor" else "#FF4D4F"};'>
                    <h4 style='color:{"#059669" if pred_class == "No Tumor" else "#B91C1C"}; margin-top:0;'>Recommended Actions</h4>
                    <ul style='color:#4B5563; font-size:0.95rem; margin-bottom:0; padding-left:20px;'>
                        <li style='margin-bottom:8px;'>Consult with a neurologist for clinical correlation.</li>
                        <li style='margin-bottom:8px;'>Verify with higher resolution MRI if available.</li>
                        <li>Use the <b>AI Assistant</b> for further queries about {pred_class}.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# 🤖 PAGE: AI MEDICAL ASSISTANT
# ==========================================

elif menu == "AI Medical Assistant":
    st.markdown("<h1>🩺 AI Medical Assistant</h1>", unsafe_allow_html=True)
    st.markdown("Ask questions about brain tumors, MRI scans, or medical report terminology.", unsafe_allow_html=True)
    
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your NeuroVision medical assistant. How can I help you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about MRI results or tumor types..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if check_ollama_status():
                # Retrieve context
                context_data = st.session_state.get('last_results', None)
                if context_data:
                    context_str = f"Specific Patient Context: The MRI scan analyzed shows a high probability ({context_data['confidence']*100:.1f}%) of {context_data['class']}."
                else:
                    context_str = "Context: No specific MRI scan has been uploaded/analyzed yet."
                
                medical_prompt = f"You are a professional medical AI assistant for brain tumor diagnosis. {context_str} User asks: {prompt}. Keep answers concise, professional, and medical."
                
                # Stream response
                response = st.write_stream(query_ollama_stream(medical_prompt))
            else:
                response = "⚠️ **System Error**: Ollama AI engine is not connected.\n\nPlease ensure Ollama is running (`ollama serve`) and you have the model pulled (`ollama pull phi`)."
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})




# ==========================================
# 📚 PAGE: EDUCATION LIBRARY
# ==========================================
elif menu == "Education Library":
    # top banner and intro
    st.markdown("<h1>📚 Knowledge Library</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; font-size:1.1rem; margin-bottom:1.5rem;'>Browse concise descriptions of common brain tumor types. Click a card to learn more.</p>", unsafe_allow_html=True)
    
    info_data = {
        "Glioma": "Gliomas are tumors that occur in the brain and spinal cord. They begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function.",
        "Meningioma": "A meningioma is a tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign).",
        "Pituitary": "Pituitary tumors are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too many hormones causing other problems.",
    }

    # emoji icons for each type to give visual cue
    icons = {
        "Glioma": "🧠",
        "Meningioma": "🧷",
        "Pituitary": "🧬",
    }

    cols = st.columns(len(info_data))
    for col, (title, desc) in zip(cols, info_data.items()):
        with col:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center; cursor:pointer; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.03)'" onmouseout="this.style.transform='scale(1)'">
                <div style="font-size:3rem; margin-bottom:0.5rem;">{icons.get(title, '📘')}</div>
                <h3 style='color:#6C63FF; margin-top:0;'>{title}</h3>
                <p style='color:#4B5563; line-height:1.4; font-size:0.95rem;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 🦶 FOOTER
# ==========================================
st.markdown("""
<div style='text-align:center; margin-top:50px; padding:20px; color:#9CA3AF; font-size:0.8rem; border-top:1px solid #E5E7EB;'>
    NeuroVision AI Diagnostic System &copy; 2026<br>
    <span style='color:#D1D5DB;'>AI-assisted tool. Results should be verified by medical professionals.</span>
</div>
""", unsafe_allow_html=True)

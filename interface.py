import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 1. Configurações de Página e Estilo Hospitalar de Alto Contraste
st.set_page_config(page_title="Cytoterma Clinical", layout="wide")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    }
    
    h1 {
        color: #0F4C81 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }
    
    h3 {
        color: #0F4C81 !important;
        font-weight: 600;
        margin-top: 20px !important;
    }

    /* Banners de Resultado (If/Else Visual) - Cores Hospitalares Puro Contraste */
    .result-banner {
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 25px;
        border: 3px solid;
        text-align: center;
    }
    .sick-banner {
        background-color: #FFF5F5 !important;
        border-color: #E53E3E !important; /* Vermelho Alerta */
        color: #C53030 !important;
    }
    .normal-banner {
        background-color: #EFF6FF !important;
        border-color: #1E40AF !important; /* Azul Clínico */
        color: #1E3A8A !important;
    }
    
    label p {
        font-size: 1.1rem !important;
        color: #0F4C81 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📋 SISTEMA DE TRIAGEM TÉRMICA — CYTOTERMA")
st.markdown("<p style='font-size:1.15rem; color:#475569; margin-top:-10px;'>Análise computacional de assimetrias vasculares e monitoramento tecidual infravermelho.</p>", unsafe_allow_html=True)
st.markdown("---")

# 2. Núcleo de Carga do Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def carregar_modelo():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("models/melhor_modelo_finetuned.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    model = carregar_modelo()
    st.sidebar.success("🤖 SISTEMA ONLINE")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar o núcleo de IA: {e}")

st.sidebar.header("📄 Identificação")
st.sidebar.text_input("Prontuário / ID Paciente", "PX-2026-8831")
st.sidebar.selectbox("Sensor Térmico", ["Flir E76 (DMR-IR)", "Flir T530", "Outro Modelo"])

# 3. Área de Upload
uploaded_file = st.file_uploader("📂 Arraste ou selecione o termograma para triagem imediata:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    t = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = t(image).unsqueeze(0).to(device)
    
    img_tensor.requires_grad = True
    output = model(img_tensor)
    classe = torch.argmax(output).item()
    
    model.zero_grad()
    output[0, classe].backward()
    grads = img_tensor.grad.abs()[0].cpu().numpy()
    heatmap = np.mean(grads, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0: 
        heatmap /= heatmap.max()
        
    st.markdown("### 🔍 Avaliação do Diagnóstico")
    
    # Exibição do Banner Direto baseado na Classe
    if classe == 0:
        st.markdown(f"""
            <div class="result-banner sick-banner">
                <span style="font-size: 2rem; font-weight: 800; letter-spacing: 1px;">🚨 ANÁLISE: SICK (ANÔMALO)</span>
                <p style="font-size: 1.25rem; margin-top: 10px; font-weight: 500;">Gradientes térmicos e assimetrias vasculares fora dos padrões normativos detectados pela IA.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-banner normal-banner">
                <span style="font-size: 2rem; font-weight: 800; letter-spacing: 1px;">✅ ANÁLISE: NORMAL</span>
                <p style="font-size: 1.25rem; margin-top: 10px; font-weight: 500;">Distribuição térmica simétrica e homogênea dentro dos limites biológicos normativos.</p>
            </div>
        """, unsafe_allow_html=True)

    # Exibição das Imagens Lado a Lado
    st.markdown("### 📊 Mapeamento e Termografia")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Termograma Original do Paciente", use_container_width=True)
        
    with col2:
        img_np = np.array(image.resize((224, 224))) / 255.0
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        overlap = cv2.addWeighted(np.float32(img_np), 0.6, np.float32(heatmap_colored), 0.4, 0)
        
        st.image(overlap, caption="Zonas de Atenção da Rede Neural (GradCAM)", use_container_width=True)

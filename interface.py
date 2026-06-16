import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 1. Configuração de Página
st.set_page_config(page_title="Cytoterma Clinical", layout="wide")

# CSS para polimento extra e letras grandes
st.markdown("""
    <style>
    /* Forçar contraste em componentes que o tema nativo às vezes ignora */
    .stTextInput input, .stSelectbox div {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    label p {
        font-size: 1.1rem !important;
        color: #0F4C81 !important;
        font-weight: 600 !important;
    }
    /* Estilo dos Banners de Resultado */
    .result-banner {
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 25px;
        border: 2px solid;
    }
    .sick-banner {
        background-color: #FFF5F5;
        border-color: #E53E3E;
        color: #C53030;
    }
    .normal-banner {
        background-color: #F0FFF4;
        border-color: #38A169;
        color: #2F855A;
    }
    /* Legenda Hospitalar Profissional */
    .legenda-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 30px;
        border-radius: 8px;
        font-size: 1.3rem !important;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📋 SISTEMA DE TRIAGEM TÉRMICA — CYTOTERMA")
st.markdown("---")

# --- Lógica do Modelo ---
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
    st.sidebar.error(f"Erro: {e}")

# Sidebar
st.sidebar.header("📄 Identificação")
st.sidebar.text_input("Prontuário / ID", "PX-2026-8831")
st.sidebar.selectbox("Sensor Térmico", ["Flir E76", "Flir T530", "Outro"])

# Upload
uploaded_file = st.file_uploader("📂 Arraste o exame térmico (JPG/PNG):", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Transformação e Predição
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tensor = t(image).unsqueeze(0).to(device)
    
    # Gerar predição simplificada
    output = model(img_tensor)
    classe = torch.argmax(output).item()
    
    st.markdown("### 🔍 Resultado da Análise")
    
    # Lógica IF / ELSE para exibição do Banner Médico
    if classe == 0:
        st.markdown(f"""<div class="result-banner sick-banner">
            <b style="font-size: 1.6rem;">🚨 ALERTA: CLASSE 0 — SICK </b><br>
            Evidências de hipertermia tecidual e assimetria vascular. Recomenda-se exame clínico urgente.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-banner normal-banner">
            <b style="font-size: 1.6rem;">✅ DIAGNÓSTICO: CLASSE 1 — NORMAL</b><br>
            Padrão térmico homogêneo e simétrico detectado. Sem pontos quentes suspeitos.
        </div>""", unsafe_allow_html=True)

    # Exibição das Imagens em Abas
    t1 = st.tabs(["📊 Visualização do Exame"])
    
    with t1:
        c1, c2 = st.columns(2)
        c1.image(image, caption="Original", use_container_width=True)
        

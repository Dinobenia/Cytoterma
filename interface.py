import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 1. Configurações da Página e Injeção de Design Hospitalar de Alto Contraste
st.set_page_config(page_title="Cytoterma - Clinical Diagnostic", layout="wide")

st.markdown("""
    <style>
    /* 1. FORÇAR FUNDO BRANCO E TEXTO ESCURO GLOBAL */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    }
    
    /* 2. RESOLVER DEFINITIVAMENTE AS LETRAS ESCONDIDAS EM INPUTS E CAIXAS DE TEXTO */
    input, select, textarea, div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
    }
    
    /* Garantir que o texto digitado e selecionado fique preto nítido */
    div[data-testid="stMarkdownContainer"] p, p, span, label, small {
        color: #1E293B !important;
    }
    
    /* Forçar a cor dos rótulos dos campos (Labels) para Azul Clínico */
    label[data-testid="stWidgetLabel"] p {
        color: #0F4C81 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    /* Forçar o texto interno de caixas de input preenchidas (como o ID do Paciente) */
    input[data-testid="stTextInputBaseInput"] {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Forçar o texto dentro do seletor (Dropdown) */
    div[data-baseweb="select"] div {
        color: #1E293B !important;
    }
    
    /* 3. CORREÇÃO DO FILE UPLOADER (ÁREA DE DROP) */
    [data-testid="stFileUploader"] {
        background-color: #F8FAFC !important;
        border: 2px dashed #0F4C81 !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span, 
    [data-testid="stFileUploader"] div, 
    [data-testid="stFileUploader"] small {
        color: #1E293B !important;
        font-size: 1.1rem !important;
    }
    
    /* 4. CORREÇÃO DAS ABAS (TABS) */
    button[data-baseweb="tab"] {
        background-color: #F1F5F9 !important;
        color: #475569 !important;
        border: 1px solid #E2E8F0 !important;
        border-bottom: none !important;
        padding: 12px 24px !important;
        font-size: 1.15rem !important;
    }
    button[aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F4C81 !important; 
        border-top: 4px solid #0F4C81 !important;
        font-weight: bold !important;
    }
    
    /* 5. CORREÇÃO DA BARRA LATERAL (SIDEBAR) */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F8FAFC !important;
        border-right: 3px solid #E2E8F0 !important;
    }
    
    /* 6. TÍTULOS E BANNERS HOSPITALARES (IF/ELSE) */
    h1 {
        color: #0F4C81 !important; 
        font-weight: 700 !important;
        font-size: 2.3rem !important;
    }
    h2, h3 {
        color: #0F4C81 !important;
        font-weight: 600 !important;
    }
    
    .status-card-sick {
        background-color: #FFF5F5 !important;
        border: 2px solid #E53E3E !important;
        border-left: 10px solid #E53E3E !important;
        padding: 25px;
        border-radius: 6px;
    }
    .status-card-normal {
        background-color: #F0FFF4 !important;
        border: 2px solid #38A169 !important;
        border-left: 10px solid #38A169 !important;
        padding: 25px;
        border-radius: 6px;
    }
    
    /* LEGENDA AMPLIADA */
    .legenda-container {
        background-color: #F8FAFC !important;
        border: 2px solid #E2E8F0 !important;
        padding: 30px;
        border-radius: 8px;
    }
    .legenda-item {
        font-size: 1.25rem !important;
        line-height: 1.6 !important;
        margin-bottom: 18px;
        color: #1E293B !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Cabeçalho Clínico
st.title("📋 SISTEMA DE TRIAGEM TÉRMICA MAMÁRIA — CYTOTERMA")
st.markdown("<p style='font-size:1.2rem; color:#334155; font-weight:500;'>Suporte de Inteligência Artificial para análise de assimetrias vasculares e monitoramento térmico tecidual infravermelho.</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Carregar o Modelo Salvo
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
    st.sidebar.markdown("<h2 style='color:#38A169; margin-top:10px;'>🟢 STATUS: ATIVO</h2>", unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error(f"Erro ao carregar modelo: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Identificação do Exame")
st.sidebar.text_input("ID do Paciente", "PX-2026-8831")
st.sidebar.selectbox("Equipamento Utilizado", ["Flir E76 (DMR-IR)", "Flir T530", "Outro Modelo"])

# 4. Transformações de Imagem
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def gerar_saliencia(model, img_tensor):
    img_tensor.requires_grad = True
    output = model(img_tensor)
    class_idx = torch.argmax(output)
    model.zero_grad()
    output[0, class_idx].backward()
    grads = img_tensor.grad.abs()[0].cpu().numpy()
    heatmap = np.mean(grads, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0: heatmap /= heatmap.max()
    return heatmap, class_idx.item()

# 5. Upload do Arquivo
uploaded_file = st.file_uploader("Selecione ou arraste o termograma para triagem clínica:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform_test(image).unsqueeze(0).to(device)
    
    with st.spinner("Analisando matriz de densidade térmica..."):
        heatmap, classe_resultado = gerar_saliencia(model, input_tensor)
    
    st.markdown("---")
    
    # 6. LÓGICA IF / ELSE — Banners Médicos de Alto Contraste
    if classe_resultado == 0:
        st.markdown(f"""
            <div class="status-card-sick">
                <div style="font-size: 1.45rem; font-weight: bold; color: #C53030; margin-bottom: 5px;">🚨 ANÁLISE DE TRIAGEM: CLASSE 0 — SICK (ANÔMALO)</div>
                <div style="font-size: 1.15rem; color: #1E293B; line-height: 1.5;">Foram identificadas assimetrias térmicas significativas e pontos focais de hipertermia. Recomenda-se encaminhamento prioritário para correlação clínica e mamografia diagnóstica.</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="status-card-normal">
                <div style="font-size: 1.45rem; font-weight: bold; color: #2F855A; margin-bottom: 5px;">✅ ANÁLISE DE TRIAGEM: CLASSE 1 — NORMAL</div>
                <div style="font-size: 1.15rem; color: #1E293B; line-height: 1.5;">Distribuição de temperatura homogênea e simétrica entre as mamas. Sem evidências de focos de radiação infravermelha anômalos detectados pela IA.</div>
            </div>
        """, unsafe_allow_html=True)
        
    # 7. Organização em Abas Visíveis
    tab_analise, tab_legenda = st.tabs(["📊 Imagens e Mapeamento", "📑 Diretrizes e Critérios do Laudo"])
    
    with tab_analise:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Termograma Original do Paciente", use_container_width=True)
        with col2:
            img_np = np.array(image.resize((224, 224))) / 255.0
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
            overlap = cv2.addWeighted(np.float32(img_np), 0.6, np.float32(heatmap_colored), 0.4, 0)
            st.image(overlap, caption="Mapa de Saliência de Atenção (Foco da IA)", use_container_width=True)
            
    with tab_legenda:
        st.markdown("""
        <div class="legenda-container">
            <h3 style="color:#0F4C81; font-size:1.5rem; margin-top:0; font-weight: bold;">📖 Critérios Técnicos para Interpretação do Laudo</h3>
            <hr style='border-color: #CBD5E1; margin-bottom: 20px;'>
            <div class="legenda-item">
                <b style="color: #E53E3E;">🔴 Padrão Patológico (Sick):</b> 
                Caracteriza-se por um gradiente térmico elevado (Δt ≥ 1°C) entre áreas homólogas, zonas circulares de retenção de calor ("hot spots") ou hipervascularização. No Mapa de Saliência, as regiões mapeadas em <b>tons quentes (vermelho e laranja)</b> revelam onde a IA fixou os parâmetros de decisão para gerar o alerta.
            </div>
            <div class="legenda-item">
                <b style="color: #38A169;">🔵 Padrão Fisiológico (Normal):</b> 
                Exibe termocromia e redes vasculares perfeitamente simétricas bilateralmente, com dissipação térmica uniforme e sem assimetrias discrepantes.
            </div>
            <p style='font-size: 0.95rem; color: #64748B; margin-top: 25px; font-style: italic;'>
                *Isenção de responsabilidade: Este software é uma ferramenta complementar de triagem prototípica baseada em redes neurais convolucionais (Visão Computacional) e não substitui o diagnóstico médico soberano, biópsias ou exames de imagem tradicionais.*
            </p>
        </div>
        """, unsafe_allow_html=True)

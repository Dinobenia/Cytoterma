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
    /* Forçar fundo branco e remover o tema escuro padrão do Streamlit */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important;
        color: #1A202C !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    }
    
    /* Correção das Abas (Tabs) para não ficarem pretas/escondidas */
    button[data-baseweb="tab"] {
        background-color: #F8F9FA !important;
        color: #4A5568 !important;
        border: 1px solid #E2E8F0 !important;
        border-bottom: none !important;
        padding: 10px 20px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    button[aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F4C81 !important; /* Azul Clínico */
        border-top: 3px solid #0F4C81 !important;
        font-weight: bold !important;
    }
    
    /* Correção do Box de Upload de Arquivos */
    [data-testid="stFileUploader"] {
        background-color: #F8F9FA !important;
        border: 2px dashed #0F4C81 !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] dropzone {
        color: #1A202C !important;
    }
    
    /* Ajustar estilo da barra lateral */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F1F5F9 !important;
        border-right: 2px solid #E2E8F0 !important;
    }
    
    /* Customização de Títulos */
    h1 {
        color: #0F4C81 !important; /* Azul Clínico Principal */
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }
    h2, h3 {
        color: #1E3A8A !important; /* Azul de Contraste */
        font-weight: 600 !important;
    }
    
    /* Cards Condicionais (If/Else Visual) */
    .status-card-sick {
        background-color: #FFF5F5 !important;
        border: 2px solid #E53E3E !important;
        border-left: 8px solid #E53E3E !important;
        padding: 22px;
        border-radius: 6px;
        margin-bottom: 25px;
    }
    .status-card-normal {
        background-color: #F0FFF4 !important;
        border: 2px solid #38A169 !important;
        border-left: 8px solid #38A169 !important;
        padding: 22px;
        border-radius: 6px;
        margin-bottom: 25px;
    }
    .card-title {
        font-size: 1.4rem !important;
        font-weight: bold !important;
        margin-bottom: 8px;
    }
    .card-desc {
        font-size: 1.1rem !important;
        color: #2D3748 !important;
        line-height: 1.5;
    }
    
    /* Aumento drástico do tamanho e legibilidade da LEGENDA */
    .legenda-container {
        background-color: #F8F9FA !important;
        border: 1px solid #E2E8F0 !important;
        padding: 25px;
        border-radius: 8px;
        color: #1A202C !important;
    }
    .legenda-item {
        font-size: 1.2rem !important; /* Letra maior para leitura médica */
        line-height: 1.6 !important;
        margin-bottom: 15px;
        color: #2D3748 !important;
    }
    .legenda-titulo-item {
        font-weight: bold !important;
        color: #0F4C81 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Cabeçalho Clínico
st.title("📋 SISTEMA DE TRIAGEM TÉRMICA MAMÁRIA — CYTOTERMA")
st.markdown("<p style='font-size:1.15rem; color:#4A5568;'>Suporte de Inteligência Artificial para análise de assimetrias vasculares e monitoramento térmico tecidual infravermelho.</p>", unsafe_allow_html=True)
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
    st.sidebar.markdown("<h3 style='color:#38A169;'>🟢 STATUS: ATIVO</h3>", unsafe_allow_html=True)
    st.sidebar.caption("Rede Neural ResNet-18 carregada em memória.")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar modelo: {e}")

# Campos na barra lateral
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
uploaded_file = st.file_uploader("Arraste e solte o termograma aqui ou clique para selecionar o arquivo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform_test(image).unsqueeze(0).to(device)
    
    with st.spinner("Analisando gradientes de temperatura..."):
        heatmap, classe_resultado = gerar_saliencia(model, input_tensor)
    
    st.markdown("---")
    
    # 6. LÓGICA IF / ELSE — Renderização dos Resultados Clínicos
    if classe_resultado == 0:
        st.markdown(f"""
            <div class="status-card-sick">
                <div class="card-title" style="color: #E53E3E;">🚨 DIAGNÓSTICO COMPUTACIONAL: CLASSE 0 — SICK (ANÔMALO)</div>
                <div class="card-desc">Foram identificadas assimetrias térmicas significativas e focos de hipertermia localizada. Recomenda-se encaminhamento prioritário para correlação clínica e mamografia diagnóstica.</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="status-card-normal">
                <div class="card-title" style="color: #38A169;">✅ DIAGNÓSTICO COMPUTACIONAL: CLASSE 1 — NORMAL</div>
                <div class="card-desc">Distribuição de temperatura homogênea e simétrica entre os quadrantes mamários. Ausência de pontos quentes infravermelhos isolados detectados pela IA.</div>
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
            <h3>📖 Critérios Técnicos para Interpretação do Laudo</h3>
            <hr style='border-color: #E2E8F0; margin-bottom: 20px;'>
            <div class="legenda-item">
                <span class="legenda-titulo-item">🔴 Padrão Patológico (Sick):</span> 
                Caracteriza-se por um gradiente térmico elevado (Δt ≥ 1°C) entre áreas homólogas das mamas, zonas circulares de retenção de calor ("hot spots") ou vascularização anômala peritumoral. No Mapa de Saliência ao lado, as regiões mapeadas em <b>tons quentes (vermelho e laranja)</b> mostram exatamente onde a IA fixou os parâmetros de peso para gerar o alerta de anomalia.
            </div>
            <div class="legenda-item">
                <span class="legenda-titulo-item">🔵 Padrão Fisiológico (Normal):</span> 
                Exibe termocromia e padrões vasculares simétricos bilateralmente, com dissipação de calor uniforme pelos tecidos moles e sem pontos focais de emissão infravermelha discrepantes.
            </div>
            <p style='font-size: 0.9rem; color: #718096; margin-top: 25px; font-style: italic;'>
                *Aviso: Este software é uma ferramenta complementar de triagem prototípica baseada em redes neurais convolucionais (Visão Computacional) e não substitui o diagnóstico médico soberano, biópsias ou exames de imagem tradicionais.*
            </p>
        </div>
        """, unsafe_allow_html=True)

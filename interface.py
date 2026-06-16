import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 1. Configurações da Página e Injeção de Design Hospitalar (Clean/Helvetica)
st.set_page_config(page_title="Cytoterma - Clinical Diagnostic", layout="wide")

st.markdown("""
    <style>
    /* Forçar fundo branco e tipografia estilo Helvetica */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        font-family: "Helvetica Now", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    }
    
    /* Ajustar estilo da barra lateral */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E9ECEF !important;
    }
    
    /* Customização de Títulos e Textos */
    h1, h2, h3 {
        color: #0F4C81 !important; /* Azul Clínico */
        font-weight: 600 !important;
    }
    
    /* Estilização dos Cards Condicionais (If/Else Visual) */
    .status-card-sick {
        background-color: #FFF5F5;
        border-left: 5px solid #E53E3E;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 25px;
    }
    .status-card-normal {
        background-color: #F0FFF4;
        border-left: 5px solid #38A169;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 25px;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #4A5568;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Cabeçalho Estilo Prontuário Médico
st.title("📋 SISTEMA DE TRIAGEM TÉRMICA MAMÁRIA — CYTOTERMA")
st.write("Suporte de Inteligência Artificial para análise de assimetrias e anomalias vasculares teciduais.")
st.markdown("---")

# 3. Carregar o Modelo Salvo (Caminho corrigido para a nuvem)
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
    st.sidebar.markdown("### 🟢 Status do Sistema")
    st.sidebar.caption("Núcleo de Processamento ResNet-18 Ativo.")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar modelo: {e}")

# Metadados simulados na barra lateral para usabilidade
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Informações do Exame")
st.sidebar.text_input("ID do Paciente", "PX-2026-8831")
st.sidebar.selectbox("Equipamento", ["Flir E76 (DMR-IR Calibrated)", "Flir T530", "Outro"])

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

# 5. Upload do Arquivo (Área de Drop)
uploaded_file = st.file_uploader("Arraste ou selecione o arquivo DICOM/Imagem Térmica convertida...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform_test(image).unsqueeze(0).to(device)
    
    with st.spinner("Processando matriz de densidade térmica..."):
        heatmap, classe_resultado = gerar_saliencia(model, input_tensor)
    
    # 6. LÓGICA IF / ELSE — Renderização dos Resultados Customizados
    if classe_resultado == 0:
        # Bloco SICK
        st.markdown(f"""
            <div class="status-card-sick">
                <div class="card-title" style="color: #C53030;">🚨 ANÁLISE COMPUTAÇÃO: CLASSE 0 — SICK</div>
                <div class="card-desc">Foram detectadas assimetrias térmicas significativas ou focos de hipertermia localizada. Recomenda-se correlação imediata com achados mamográficos e clínicos.</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Bloco NORMAL
        st.markdown(f"""
            <div class="status-card-normal">
                <div class="card-title" style="color: #2F855A;">✅ ANÁLISE COMPUTACIONAL: CLASSE 1 — NORMAL</div>
                <div class="card-desc">Distribuição de temperatura simétrica dentro dos padrões normativos de homogeneidade tecidual. Sem evidências de pontos quentes anômalos.</div>
            </div>
        """, unsafe_allow_html=True)
        
    # 7. Organização Visual em Abas (Melhoria de Usabilidade)
    tab_analise, tab_legenda = st.tabs(["📊 Imagens do Laudo", "📑 Critérios de Análise e Legenda"])
    
    with tab_analise:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Termograma Original (Input)", use_container_width=True)
        with col2:
            img_np = np.array(image.resize((224, 224))) / 255.0
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
            overlap = cv2.addWeighted(np.float32(img_np), 0.6, np.float32(heatmap_colored), 0.4, 0)
            st.image(overlap, caption="Mapa de Saliência GradCAM (Regiões de Interesse)", use_container_width=True)
            
    with tab_legenda:
        st.markdown("""
        ### Diretrizes para Interpretação Médica
        
        * **Padrão Patológico (Sick):** Caracteriza-se por gradientes térmicos delta superiores a 1°C entre pontos homólogos das mamas, zonas localizadas de captação de calor ("hot spots") ou hipervascularização peritumoral visível. As regiões que tendem ao **vermelho** no mapa indicam o foco de decisão da rede neural.
        * **Padrão Fisiológico (Normal):** Apresenta termocromia simétrica bilateral, ausência de ramificações vasculares isoladas com emissão de calor infravermelho discrepante.
        
        <p style='font-size: 0.85rem; color: #718096; margin-top: 20px;'>
        *Isenção de responsabilidade: Este software é classificado como uma ferramenta experimental de triagem (Classificação Binária via Visão Computacional) e serve como complemento diagnóstico prototípico.*
        </p>
        """, unsafe_allow_html=True)

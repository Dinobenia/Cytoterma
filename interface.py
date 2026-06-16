import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configurações da página
st.set_page_config(page_title="Cytoterma - Análise Térmica", layout="wide")
st.title("🌡️ Cytoterma - Sistema de Triagem de Termografia Mamária")
st.write("Carregue a imagem térmica do paciente para análise imediata da inteligência artificial.")

# Carregando o modelo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def carregar_modelo():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    # Carrega os pesos gerados no treino
    model.load_state_dict(torch.load("models/melhor_modelo_finetuned.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    model = carregar_modelo()
    st.sidebar.success("🤖 Modelo IA carregado com sucesso!")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar modelo: {e}")

# Ajustando imagem
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapa de saliência
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
    return heatmap, class_idx

# Interface Visual
uploaded_file = st.file_uploader("Selecione uma imagem térmica (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    input_tensor = transform_test(image).unsqueeze(0).to(device)
    
    # Predição e o mapa de saliência
    with st.spinner("Analisando tecidos térmicos..."):
        heatmap, classe_resultado = gerar_saliencia(model, input_tensor)
    
    # Resultado
    st.markdown("---")
    if classe_resultado == 0:
        st.error("### 🚨 Resultado da Análise: SICK (Sinais de Anomalia Térmica Detectados)")
    else:
        st.success("### ✅ Resultado da Análise: NORMAL (Padrão Térmico Saudável)")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Imagem Térmica Original", use_container_width=True)
        
    with col2:

        img_np = np.array(image.resize((224, 224))) / 255.0
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        
        overlap = cv2.addWeighted(np.float32(img_np), 0.6, np.float32(heatmap_colored), 0.4, 0)
        st.image(overlap, caption="Mapa de Saliência (Áreas de Foco da IA)", use_container_width=True)

    # Legenda explicativa médica
    st.markdown("### 📑 Legenda e Critérios de Análise")
    st.info("""
    * **O que define o diagnóstico de 'Sick'?** A presença de assimetrias térmicas acentuadas, pontos quentes focais (hipertermia) ou padrões vasculares anômalos na mama. No Mapa de Saliência, as regiões que tendem ao **vermelho e amarelo** mostram exatamente onde a IA detectou a maior variação térmica suspeita.
    * **O que define o diagnóstico de 'Normal'?** Distribuição térmica simétrica e homogênea entre os tecidos mamários, sem focos isolados de calor.
    * *Aviso: Este sistema é uma ferramenta de suporte à triagem e não substitui exames clínicos e mamografias tradicionais.*
    """)

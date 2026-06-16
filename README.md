# 📋 Cytoterma — Sistema de Triagem de Termografia Mamária

## 🚀 Objetivo do Projeto
Este sistema inteligente foi desenvolvido para realizar a classificação binária e triagem automática de imagens de termografia mamária infravermelha, auxiliando profissionais da saúde na detecção precoce de assimetrias térmicas e anomalias teciduais associadas a patologias vasculares.

## 🛠️ Tecnologias Utilizadas
* **Linguagem Principal:** Python 3.12
* **Deep Learning & Visão Computacional:** PyTorch, Torchvision
* **Interface de Usuário & Deploy:** Streamlit Cloud
* **Processamento de Imagem:** OpenCV, Pillow, NumPy
* **Versionamento:** Git & GitHub

## 💻 Como Executar Localmente
1. Clone o repositório: `git clone https://github.com/Dinobenia/Cytoterma.git`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute a interface: `streamlit run interface.py`

## 🧠 Descrição do Modelo de IA
O núcleo do sistema utiliza a arquitetura **ResNet-18** com técnicas de Fine-Tuning. O modelo foi treinado de forma estritamente binária (isolando dados redundantes) por 25 épocas utilizando taxa de aprendizado diferencial nos blocos convolucionais profundos. Contém filtros heurísticos para descarte de imagens incompatíveis ou de baixa confiança.

## 👥 Autores do Projeto
* **Beatriz Caixeta Grosara e Evelyn Barboza** — Estudantes de Tecnologia em Ciência de Dados (Faculdade SENAC DF)

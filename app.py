# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES INICIAIS
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random

# --- CONFIGURAÇÃO DE REPRODUTIBILIDADE (SEMENTE ALEATÓRIA) ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- CONFIGURAÇÃO DE DISPOSITIVO (GPU OU CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Ambiente configurado. Usando dispositivo: {device}")


# ==============================================================================
# 2. CONFIGURAÇÕES DE TRANSFORMAÇÕES (DATA AUGMENTATION)
# ==============================================================================
# Transformações para o conjunto de treino (com augmentations para evitar overfitting)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformações para validação e teste (apenas redimensionamento e normalização)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==============================================================================
# 3. MAPEAMENTO DO DATASET E CRIAÇÃO DOS LOADERS
# ==============================================================================
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# Caminho ajustado para o novo dataset DMR-IR do Kaggle
data_dir = "/kaggle/input/breast-cancer-detection-using-thermography-dmr-ir"

# Carregando o dataset estruturado em pastas
full_dataset = datasets.ImageFolder(data_dir, transform=None)

# Mapeando e filtrando as classes do problema
valid_classes = ['Sick', 'normal']
indices = [full_dataset.class_to_idx[c] for c in valid_classes]
filtered_samples = [s for s in full_dataset.samples if s[1] in indices]

full_dataset.samples = filtered_samples
full_dataset.targets = [s[1] for s in filtered_samples]

class_map = {indices[0]: 0, indices[1]: 1}
full_dataset.samples = [(p, class_map[l]) for p, l in full_dataset.samples]
full_dataset.targets = [class_map[l] for l in full_dataset.targets]
target_names = ['Doente (Classe 0)', 'Saudável (Classe 1)']

# Divisão do dataset (70% Treino, 10% Validação, 20% Teste)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])

train_dataset = TransformedSubset(train_subset, transform=transform_train)
val_dataset = TransformedSubset(val_subset, transform=transform_test)
test_dataset = TransformedSubset(test_subset, transform=transform_test)

# Criação dos DataLoaders para alimentar a rede neural
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"[INFO] Dataset processado com sucesso!")
print(f"Total: {total_size} imagens -> Treino: {train_size} | Val: {val_size} | Teste: {test_size}")


# ==============================================================================
# 4. CONFIGURAÇÃO DA ARQUITETURA DO MODELO (FINE-TUNING RESNET-18)
# ==============================================================================
# Carregando a ResNet-18 com pesos pré-treinados padrão do ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 1. Congelar todos os parâmetros do modelo primeiro
for param in model.parameters():
    param.requires_grad = False

# 2. Descongelar as camadas finais (layer3 e layer4) para o ajuste fino
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# 3. Substituir a última camada linear (classificador) para o nosso problema (2 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Enviar o modelo para a GPU ou CPU
model = model.to(device)


# ==============================================================================
# 5. CRITÉRIO DE PERDA E OTIMIZADOR DIFERENCIAL
# ==============================================================================
criterion = nn.CrossEntropyLoss()

# Criando grupos de parâmetros com taxas de aprendizado (Learning Rates) diferentes
params_to_optimize = [
    {'params': model.fc.parameters(), 'lr': 0.001},         # Camada nova aprende mais rápido
    {'params': model.layer3.parameters(), 'lr': 0.0001},    # Camada profunda ajusta devagar
    {'params': model.layer4.parameters(), 'lr': 0.0001}     # Camada profunda ajusta devagar
]

optimizer = optim.Adam(params_to_optimize)

print("[INFO] Modelo ResNet-18 e Otimizador Diferencial configurados com sucesso!")

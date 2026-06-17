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

# --- CONFIGURAÇÃO DE REPRODUTIBILIDADE ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Ambiente configurado. Usando dispositivo: {device}")

# ==============================================================================
# 2. CONFIGURAÇÕES DE TRANSFORMAÇÕES (DATA AUGMENTATION)
# ==============================================================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

# CAMINHO CORRIGIDO BASEADO NO SEU DIRETÓRIO
data_dir = "/kaggle/input/datasets/BCD_Dataset"

full_dataset = datasets.ImageFolder(data_dir, transform=None)

valid_classes = ['Sick', 'normal']
indices = [full_dataset.class_to_idx[c] for c in valid_classes]
filtered_samples = [s for s in full_dataset.samples if s[1] in indices]

full_dataset.samples = filtered_samples
full_dataset.targets = [s[1] for s in filtered_samples]

class_map = {indices[0]: 0, indices[1]: 1}
full_dataset.samples = [(p, class_map[l]) for p, l in full_dataset.samples]
full_dataset.targets = [class_map[l] for l in full_dataset.targets]
target_names = ['Doente (Classe 0)', 'Saudável (Classe 1)']

total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])

train_dataset = TransformedSubset(train_subset, transform=transform_train)
val_dataset = TransformedSubset(val_subset, transform=transform_test)
test_dataset = TransformedSubset(test_subset, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"[INFO] Dataset processado! Total: {total_size} imagens (Treino: {train_size}, Val: {val_size}, Teste: {test_size})")

# ==============================================================================
# 4. CONFIGURAÇÃO DA ARQUITETURA DO MODELO (FINE-TUNING RESNET-18)
# ==============================================================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer3.parameters(): param.requires_grad = True
for param in model.layer4.parameters(): param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# ==============================================================================
# 5. CRITÉRIO DE PERDA E OTIMIZADOR DIFERENCIAL
# ==============================================================================
criterion = nn.CrossEntropyLoss()
params_to_optimize = [
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.layer3.parameters(), 'lr': 0.0001},
    {'params': model.layer4.parameters(), 'lr': 0.0001}
]
optimizer = optim.Adam(params_to_optimize)
print("[INFO] Modelo ResNet-18 e Otimizador Diferencial configurados!")

# ==============================================================================
# 6. LOOP DE TREINAMENTO E VALIDAÇÃO
# ==============================================================================
epochs = 25
best_val_loss = float('inf')
NOME_DO_ARQUIVO = "Cytoterma/models/melhor_modelo_finetuned.pth"

print("\n[INFO] Iniciando o treinamento por 25 épocas...")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Época {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), NOME_DO_ARQUIVO)

print("[INFO] Treinamento finalizado!")

# ==============================================================================
# 7. AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (MÉTRICAS)
# ==============================================================================
print("\n[INFO] Carregando o melhor modelo para avaliação final...")
model.load_state_dict(torch.load(NOME_DO_ARQUIVO))
model.eval()

correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader: 
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nAcurácia final no conjunto de TESTE: {100 * correct / total:.2f}%")
print("\n--- Relatório de Classificação ---")
print(classification_report(all_labels, all_preds, target_names=target_names))

# ==============================================================================
# 8. EXPLICABILIDADE: MAPA DE SALIÊNCIA (XAI)
# ==============================================================================
def saliency_map(model, img_tensor):
    img_tensor.requires_grad = True
    output = model(img_tensor)
    class_idx = torch.argmax(output)
    model.zero_grad()
    output[0, class_idx].backward()
    grads = img_tensor.grad.abs()[0].cpu().numpy()
    heatmap = np.mean(grads, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0: heatmap /= heatmap.max()
    return heatmap

print("\n[INFO] Gerando mapa de saliência explicativo...")
sample_img, _ = test_dataset[0]
input_tensor = sample_img.unsqueeze(0).to(device)
heatmap = saliency_map(model, input_tensor)

inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
img_to_plot = np.clip(inv_normalize(sample_img).permute(1, 2, 0).cpu().numpy(), 0, 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(img_to_plot)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Mapa de Saliência")
plt.imshow(img_to_plot)
plt.imshow(cv2.resize(heatmap, (224, 224)), cmap='jet', alpha=0.5)
plt.axis('off')

plt.tight_layout()
plt.savefig("Cytoterma/resultado_saliencia.png")
print("[INFO] Execução completa! Imagem salva em Cytoterma/resultado_saliencia.png")

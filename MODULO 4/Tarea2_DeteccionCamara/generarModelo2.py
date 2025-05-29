#pip install torch torchvision scikit-learn matplotlib pillow tqdm
#'C:/Users/Propietario/Documents/Edgar/IA/FER-2013/both'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuración
DATA_DIR = 'C:/Users/Propietario/Documents/Edgar/IA/FER-2013/both'  # Cambiar por la ruta correcta
BATCH_SIZE = 32
IMAGE_SIZE = (48, 48)
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 5  # angry, disgust, fear, happy, neutral, sad, surprise
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformaciones para imágenes en escala de grises
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),  # Convertir a 3 canales
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # Normalización para escala de grises
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=1),  # Convertir a 3 canales
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # Normalización para escala de grises
    ]),
}

# Método para cargar el dataset
def load_dataset(data_dir):
    classes = ['angry', 'fear', 'happy', 'sad', 'surprise']
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if os.path.isfile(image_path):
                try:
                    # Verificar que es una imagen válida
                    with Image.open(image_path) as img:
                        img.verify()
                    image_paths.append(image_path)
                    labels.append(class_idx)
                except (IOError, SyntaxError) as e:
                    print(f'Error al cargar imagen {image_path}: {e}')
    
    return image_paths, labels

class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)  # Cargar imagen tal cual (puede ser RGB o escala de grises)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(EmotionClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        # Modificar la primera capa para aceptar 1 canal de entrada
        original_first_layer = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            1,  # 1 canal de entrada para escala de grises
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        
        # Congelar parámetros excepto la primera capa
        for name, param in self.base_model.named_parameters():
            if name != "conv1.weight":
                param.requires_grad = False
                
        # Reemplazar capa final
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Fase de validación
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc)
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Guardar el mejor modelo
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        print()
    
    # Reporte de clasificación
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, 
                               target_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']))
    
    # Gráficas
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    return model

if __name__ == '__main__':
    # Cargar dataset
    print("Cargando dataset...")
    image_paths, labels = load_dataset(DATA_DIR)
    print(f"Total de imágenes cargadas: {len(image_paths)}")
    
    # Dividir dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Crear datasets y dataloaders
    print("Preparando dataloaders...")
    train_dataset = EmotionDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_dataset = EmotionDataset(val_paths, val_labels, transform=data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Inicializar modelo
    print("Inicializando modelo...")
    model = EmotionClassifier(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Entrenamiento
    print("Comenzando entrenamiento...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    
    # Guardar modelo
    torch.save(model.state_dict(), 'final_model.pth')
    print('Entrenamiento completado. Modelo guardado como final_model.pth')
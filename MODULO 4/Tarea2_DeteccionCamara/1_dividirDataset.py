import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time


# Configuración
dataset_path = 'C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_redimensionado'
output_path = 'C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_split'
val_ratio = 0.2  # 20% para validación
random_seed = 42
batch_size = 32
img_size = 64  # Tamaño esperado por ResNet18

# Crear directorios train y val
os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)

# Obtener lista de clases
classes = os.listdir(dataset_path)
print(f"Clases encontradas: {classes}")

# Procesar cada clase
for cls in classes:
    # Crear subdirectorios para cada clase
    os.makedirs(os.path.join(output_path, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val', cls), exist_ok=True)
    
    # Obtener lista de imágenes
    src_dir = os.path.join(dataset_path, cls)
    images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # Dividir en train/val
    train_files, val_files = train_test_split(images, test_size=val_ratio, random_state=random_seed)
    
    # Copiar archivos a train
    for f in train_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(output_path, 'train', cls, f))
    
    # Copiar archivos a val
    for f in val_files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(output_path, 'val', cls, f))
    
    print(f"Clase {cls}: {len(train_files)} train, {len(val_files)} val")

print("División completada!")

# Transformaciones para ResNet50
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar datasets
train_dataset = datasets.ImageFolder(
    root=os.path.join(output_path, 'train'),
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(output_path, 'val'),
    transform=transform
)

# Crear DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

# Verificar
print(f"Número de clases: {len(train_dataset.classes)}")
print(f"Total imágenes de entrenamiento: {len(train_dataset)}")
print(f"Total imágenes de validación: {len(val_dataset)}")


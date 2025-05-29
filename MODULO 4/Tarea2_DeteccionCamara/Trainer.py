#pip install seaborn

import torch.multiprocessing as mp
from tqdm import tqdm  # Añade al inicio de los imports
mp.set_start_method('spawn', force=True)
from copy import deepcopy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, Dict, List, Optional

class EmotionResNetTrainer:
    def __init__(self, train_dir: str, val_dir: str, num_classes: int = 5, batch_size: int = 32):
        """
        Inicializa el entrenador de ResNet para clasificación de emociones.
        
        Args:
            train_dir (str): Directorio con imágenes de entrenamiento
            val_dir (str): Directorio con imágenes de validación
            num_classes (int): Número de clases de emociones
        """
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Inicializar transformaciones
        self._init_transforms()
        
        # Cargar datasets
        self._load_datasets()
        
        # Inicializar modelo
        self._init_model()
    
    def _init_transforms(self):
        """Inicializa las transformaciones para train y validation"""
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Nuevo
            transforms.RandomRotation(10),  # Pequeñas rotaciones
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(64+16),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_datasets(self):
        """Carga los datasets y crea los DataLoaders"""
        train_dataset = datasets.ImageFolder(
            self.train_dir,
            self.train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            self.val_dir,
            self.val_transform
        )
        
        self.class_names = train_dataset.classes
        print(f"Classes found: {self.class_names}")
        
        # Crear DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
    def _init_model(self, pretrained: bool = True):
        """Inicializa el modelo ResNet18 con fine-tuning optimizado"""
        # 1. Cargar modelo pre-entrenado
        self.model = models.resnet18(pretrained=pretrained)
        
        # 2. Congelar todas las capas inicialmente
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 3. Descongelar capas estratégicamente
        layers_to_unfreeze = ['layer3', 'layer4', 'fc']  # Capas profundas + clasificador
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
        
        # 4. Reemplazar FC con una estructura mejorada
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Regularización
            nn.Linear(num_ftrs, 256),  # Capa intermedia
            nn.ReLU(),
            nn.Linear(256, self.num_classes)  # Capa final
        )
        self.model = self.model.to(self.device)
        
        # 5. Configurar optimizador para TODOS los parámetros descongelados
        self.optimizer = optim.AdamW(  # AdamW es mejor para weight decay
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001,  # Tasa de aprendizaje inicial
            weight_decay=1e-4  # Regularización L2
        )
        
        # 6. Configurar scheduler y criterio
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        
        # 7. Criterio con balanceo de clases (opcional)
        class_weights = torch.tensor([1.0/x for x in [1419, 2320, 1536, 2185, 1483]])
        class_weights = class_weights / class_weights.sum()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
    
    def train(self, num_epochs: int = 25) -> Dict[str, List[float]]:
        """
        Entrena el modelo.
            
        Args:
            num_epochs (int): Número de épocas de entrenamiento
                
        Returns:
            Dict con historial de métricas de entrenamiento
        """
        since = time.time()
        best_model_wts = deepcopy(self.model.state_dict())
        best_acc = 0.0
            
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Fase de entrenamiento
            self.model.train()
            train_bar = tqdm(self.train_loader, desc='Training', leave=True)
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Actualizar barra de progreso
                train_bar.set_postfix({
                    'loss': f'{running_loss/((train_bar.n+1)*self.batch_size):.4f}',
                    'acc': f'{running_corrects.float()/((train_bar.n+1)*self.batch_size):.4f}'
                })
            
            # Calcular métricas de entrenamiento
            epoch_train_loss = running_loss / len(self.train_loader.dataset)
            epoch_train_acc = running_corrects.double() / len(self.train_loader.dataset)
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc.cpu().numpy())
            
            # Fase de validación
            self.model.eval()
            val_bar = tqdm(self.val_loader, desc='Validation', leave=True)
            running_loss = 0.0
            running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    val_bar.set_postfix({
                        'loss': f'{running_loss/((val_bar.n+1)*self.batch_size):.4f}',
                        'acc': f'{running_corrects.float()/((val_bar.n+1)*self.batch_size):.4f}'
                    })
            
            # Calcular métricas de validación
            epoch_val_loss = running_loss / len(self.val_loader.dataset)
            epoch_val_acc = running_corrects.double() / len(self.val_loader.dataset)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc.cpu().numpy())
            
            # Actualizar mejor modelo
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model_wts = deepcopy(self.model.state_dict())
            
            # Mostrar resumen de la época
            print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
            print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
            
            self.scheduler.step(epoch_val_acc)
        
        time_elapsed = time.time() - since
        print(f'\nEntrenamiento completo en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Mejor accuracy en validación: {best_acc:.4f}')
        
        self.model.load_state_dict(best_model_wts)
        return history
    
    def save_model(self, path: str):
        """Guarda el modelo entrenado"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Carga un modelo guardado"""
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)
        print(f"Model loaded from {path}")
    
    def evaluate(self) -> Tuple[float, float]:
        """Evalúa el modelo en el conjunto de validación"""
        self.model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        total_loss = val_loss / len(self.val_loader.dataset)
        total_acc = val_corrects.double() / len(self.val_loader.dataset)
        
        print(f'Validation Loss: {total_loss:.4f} Acc: {total_acc:.4f}')
        return total_loss, total_acc
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Realiza una predicción en una única imagen
        
        Args:
            image_tensor (torch.Tensor): Tensor de imagen ya transformado
            
        Returns:
            Tuple con (class_name, confidence)
        """
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        class_name = self.class_names[preds[0]]
        confidence = probs[0][preds[0]].item()
        
        return class_name, confidence
    
    def plot_confusion_matrix(self):
        """Grafica la matriz de confusión"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    def visualize_predictions(self, num_images: int = 6):
        """Visualiza predicciones del modelo"""
        self.model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(15, 10))
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'pred: {self.class_names[preds[j]]}\ntrue: {self.class_names[labels[j]]}')
                    img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                    
                    if images_so_far == num_images:
                        plt.tight_layout()
                        plt.show()
                        return
        
        plt.tight_layout()
        plt.show()
    
    def unfreeze_layers(self, layer_names: List[str]):
        """
        Descongela capas específicas del modelo para fine-tuning
        
        Args:
            layer_names (List[str]): Lista de nombres de capas a descongelar
                Ejemplo: ['layer4', 'fc']
        """
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                print(f"Unfreezed {name}")
    
    def set_class_weights(self, class_counts: List[int]):
        """
        Configura pesos para las clases (útil para datasets desbalanceados)
        
        Args:
            class_counts (List[int]): Conteo de imágenes por clase en orden
        """
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Class weights set successfully")
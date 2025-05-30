import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargando el modelo con la misma arquitectura (resnet 18 capas)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, 5)  # 5 clases
)

# Cargar los pesos del modelo ya entrenado
model.load_state_dict(torch.load('emotion_resnet50.pth', map_location=device))
model.eval()

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Nuevo
    transforms.RandomRotation(10),  # Pequeñas rotaciones
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Diccionario de emociones (debe coincidir con tu entrenamiento)
emotion_dict = {
    0: "Enojo",
    1: "Miedo",
    2: "Felicidad",
    3: "Tristeza",
    4: "Sorpresa"
}

def preprocesar_cara(img_cara):
    #Preprocesa el rostro para el modelo
    # Convertir BGR (OpenCV) a RGB
    rgb_img = cv2.cvtColor(img_cara, cv2.COLOR_BGR2RGB)
    # Aplicar transformaciones
    tensor_img = transform(rgb_img)
    # Añadir dimensión de batch
    return tensor_img.unsqueeze(0).to(device)

def predecir_emocion(img_cara):
    with torch.no_grad():
        inputs = preprocesar_cara(img_cara)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        return emotion_dict[preds.item()]
    
def predecir_emocion_suavizada(rostro):
    emocion = predecir_emocion(rostro)
    emocion_historial.append(emocion)
    if len(emocion_historial) > N_HISTORIAL:
        emocion_historial.pop(0)
    # Devolver la emoción más frecuente
    return max(set(emocion_historial), key=emocion_historial.count)
    
emocion_historial = []
N_HISTORIAL = 5

def detectar_rostros():
    global img_ultimo_rostro, x, y, w, h, contador_frames
    
    # Inicializar variables
    ultima_emocion = "Neutral"
    
    # Cargar clasificadores
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    perfil_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Espejo para efecto espejo
        frame = cv2.flip(frame, 1)
        frame_clean = frame.copy()
        
        # Dibujar rectángulo y texto de la última emoción
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, ultima_emocion, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Detección cada N frames
        contador_frames += 1
        if contador_frames % intervalo_deteccion == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros frontales
            frontal_faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
            
            # Si no hay rostros frontales, buscar perfiles
            perfil_faces = []
            perfil_inv_faces = []
            if len(frontal_faces) == 0:
                perfil_faces = perfil_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
                
                perfil_inv_faces = perfil_cascade.detectMultiScale(
                    cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
            
            # Combinar todas las detecciones
            total_caras = list(frontal_faces)
            for (px, py, pw, ph) in perfil_faces:
                total_caras.append((px, py, pw, ph))
            for (px, py, pw, ph) in perfil_inv_faces:
                total_caras.append((frame.shape[1]-px-pw, py, pw, ph))
            
            if len(total_caras) > 0:
                # Tomar el rostro más grande
                x, y, w, h = sorted(total_caras, key=lambda f: f[2]*f[3], reverse=True)[0]
                rostro_recortado = frame_clean[y:y+h, x:x+w]
                img_ultimo_rostro = rostro_recortado.copy()
                
                try:
                    # Predecir emoción
                    ultima_emocion = predecir_emocion_suavizada(rostro_recortado)
                    
                    # Mostrar rostro detectado
                    cv2.imshow('Rostro Detectado', rostro_recortado)
                except Exception as e:
                    print(f"Error en predicción: {e}")
                    ultima_emocion = "Error"
            
            elif img_ultimo_rostro is not None:
                # Mostrar último rostro detectado con mensaje
                cv2.putText(img_ultimo_rostro, "No detectado", (5, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow('Rostro Detectado', img_ultimo_rostro)
                img_ultimo_rostro = None
        
        # Mostrar frame principal
        cv2.imshow('Deteccion de emociones', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Variables globales
    img_ultimo_rostro = None
    x, y, w, h = 0, 0, 0, 0
    contador_frames = 0
    intervalo_deteccion = 3
    
    detectar_rostros()
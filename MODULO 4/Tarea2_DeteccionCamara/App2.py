#pip install numpy pillow keras
#pip install tensorflow
#pip install opencv-python

import cv2
from keras.models import load_model
from torchvision import transforms
from PIL import Image
import torch
import torchvision.models as models

from torchvision import transforms
from torch import nn

ruta_modelo= 'best_model.pth'
ruta_labels= 'MODULO 4/Tarea2_DeteccionCamara/labels.txt'

img_ultimo_rostro = None
x,y,w,h = 0,0,0,0 ##posiciones del rectangulo verde
contador_frames = 0
intervalo_deteccion = 3
ultima_emocion = "Neutral"

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # Normalización para escala de grises
    ]),
    'val': transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # Normalización para escala de grises
    ]),
}

# Cargar el modelo entrenado
def load_model(ruta_modelo, num_classes=5):
    model = models.resnet18(pretrained=False) 
    
      # 2. Ajusta para entrada en blanco/negro (1 canal)
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 3. Ajusta la capa fully connected para que coincida con el checkpoint
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),  # fc.0
        nn.ReLU(),
        nn.Linear(256, num_classes)            # fc.3
    )

    # 4. Carga el state_dict
    state_dict = torch.load(ruta_modelo, map_location='cpu')
    
    # 5. Elimina el prefijo 'base_model.' si existe
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('base_model.'):
            new_key = k.replace('base_model.', '')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    # 6. Carga los pesos ajustados
    model.load_state_dict(new_state_dict, strict=False)  # strict=False para ignorar discrepancias menores
    
    model.eval()

    return model


def detectar_rostros():
    global img_ultimo_rostro
    global x,y,w,h
    global contador_frames
    global intervalo_deteccion
    global ultima_emocion

    #Cosas del modelo
    model=load_model(ruta_modelo)
    labels = ['angry', 'fear', 'happy', 'sad', 'surprise']

    # Cargar el clasificador preentrenado para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    perfil_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # Inicializar la cámara (0 es la cámara por defecto)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return
    
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        
        if not ret:
            print("No se pudo recibir el frame. Saliendo...")
            break
        
        # Invertir la imagen como espejo
        frame = cv2.flip(frame, 1)
        frame_clean = frame.copy()

        # Dibujar el rectangulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Imprimir la emoción
        cv2.putText(frame, ultima_emocion, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        contador_frames += 1
        if(contador_frames % intervalo_deteccion == 0): #Es hora de detectar

            frontal_faces=[]
            perfil_faces=[]
            perfil_inv_faces=[]

            # Convertir a escala de grises (la detección funciona mejor en grises) 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros en la imagen
            frontal_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(75, 75))
            
            if(len(frontal_faces) == 0):

                perfil_faces = perfil_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(75, 75))
                
                perfil_inv_faces = perfil_cascade.detectMultiScale(
                    cv2.flip(gray, 1),  # Invertir para detectar perfil opuesto
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(75, 75))
            
            # Combinar todas las detecciones
            total_caras = []
            if len(frontal_faces) > 0:
                total_caras.extend(frontal_faces)
            if len(perfil_faces) > 0:
                total_caras.extend(perfil_faces)
            if len(perfil_inv_faces) > 0:
                # Ajustar coordenadas para caras invertidas
                for (x, y, w, h) in perfil_inv_faces:
                    total_caras.append((frame.shape[1]-x-w, y, w, h))

            if len(total_caras) > 0:
                # Ordenar los rostros por área (w*h) de mayor a menor
                total_caras = sorted(total_caras, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = total_caras[0]

                rostro_recortado = frame_clean[y:y+h, x:x+w]

                img_ultimo_rostro = rostro_recortado.copy()

                #Aplicar limpieza de imagen
                frame_deteccion = rostro_recortado.copy()
                frame_deteccion = cv2.resize(rostro_recortado, (48, 48))
                frame_deteccion_pil = Image.fromarray(cv2.cvtColor(frame_deteccion, cv2.COLOR_BGR2RGB))

                ##input
                input_tensor = data_transforms['val'](frame_deteccion_pil)
                input_batch = input_tensor.unsqueeze(0)  # Añade dimensión de batch

                #Predice
                with torch.no_grad():
                    output = model(input_batch)
                _, predicted_idx = torch.max(output, 1)
                predicted_label = labels[predicted_idx.item()]

                #mostrar prediccion
                ultima_emocion = predicted_label
                print(f"Pred: {predicted_label}")

        

                cv2.imshow('Rostro Detectado', rostro_recortado)
            else:
                if(img_ultimo_rostro is not None):
                    ultimo_rostro_txt = cv2.putText(img_ultimo_rostro, "No detectado", (5, 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.imshow('Rostro Detectado', ultimo_rostro_txt)
                    img_ultimo_rostro=None
            
        
        
        # Mostrar el frame resultante
        cv2.imshow('Deteccion de emociones', frame)
        cv2.moveWindow('Deteccion de emociones', 0,0)

        
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_rostros()

def main():
    print("iniciando")
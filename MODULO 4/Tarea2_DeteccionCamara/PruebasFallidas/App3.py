#pip install numpy pillow keras
#pip install tensorflow
#pip install opencv-python

import cv2
from keras.models import load_model
import cv2  # OpenCV para la cámara
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

ruta_modelo= 'MODULO 4/Tarea2_DeteccionCamara/modelo.tflite'
ruta_labels= 'MODULO 4/Tarea2_DeteccionCamara/labels.txt'

img_ultimo_rostro = None
x,y,w,h = 0,0,0,0 ##posiciones del rectangulo verde
contador_frames = 0
intervalo_deteccion = 3
ultima_emocion=""

def preprocess_cv2_image(cv2_img):
    """
    Process a cv2 image for prediction with your Teachable Machine model
    Input: cv2 image (BGR format by default)
    Output: Processed tensor ready for prediction
    """
    # Convert to grayscale if needed (assuming your model expects grayscale)
    if len(cv2_img.shape) == 3:
        gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = cv2_img
    
    # Resize to 48x48 (if not already that size)
    resized_img = cv2.resize(gray_img, (96, 96))
    
    # Normalize to [0, 1] (assuming your model expects this)
    normalized_img = resized_img / 255.0
    
    # Add batch and channel dimensions
    # Shape goes from (48, 48) to (1, 48, 48, 1)
    input_tensor = np.expand_dims(normalized_img, axis=0)  # Add batch dim
    input_tensor = np.expand_dims(input_tensor, axis=-1)  # Add channel dim
    
    # Convert to float32
    input_tensor = input_tensor.astype(np.float32)
    
    return input_tensor


def detectar_rostros():
    global img_ultimo_rostro
    global x,y,w,h
    global contador_frames
    global intervalo_deteccion
    global model
    global ultima_emocion
    global labels

    #Cosas del modelo tflite
    interpreter = tf.lite.Interpreter(model_path=ruta_modelo)  # Replace with your model path
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #lo que puede salir
    clases = ["Enojado", "Miedo","Feliz","Triste","Sorprendido"]

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
                frame_deteccion = cv2.resize(rostro_recortado, (96, 96))

                input_tensor = preprocess_cv2_image(frame_deteccion)
                # Make prediction
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Get prediction results
                predicted_class = np.argmax(output_data[0])
                confidence = np.max(output_data[0])

                ultima_emocion = clases[predicted_class]

                print(f"Se predice: {ultima_emocion} con confianza: {confidence:.2f}")
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
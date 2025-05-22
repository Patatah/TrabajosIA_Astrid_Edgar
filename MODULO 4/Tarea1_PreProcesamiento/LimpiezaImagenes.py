#pip install opencv-python 
import cv2
import os
import numpy as np

# Directorios
#La de menor resolución es disgust286 con 13x17
dir_origen = "C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_data"  # Carpeta raíz con subcarpetas
dir_destino = "C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_redimensionado"
os.makedirs(dir_destino, exist_ok=True)

# Tamaño objetivo (ancho, alto)
tamanio_objetivo = (60, 80) #Numero aureo? 16:10 aprox

##func normalizar brillo
def normalize_brightness(img, target_mean=128):
    """Normaliza la luminosidad de la imagen a un valor medio objetivo"""
    # Convertir a YCrCb (Y es el canal de luminancia)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Calcular el valor medio actual de Y
    current_mean = np.mean(y)
    
    # Ajustar el brillo
    y = np.clip(y * (target_mean / current_mean), 0, 255).astype(np.uint8)
    
    # Fusionar los canales y convertir de vuelta a BGR
    ycrcb = cv2.merge([y, cr, cb])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# Recorrer todas las subcarpetas
for root, dirs, files in os.walk(dir_origen):
    for file in files:
        if file.lower().endswith(('.jpg')):
            dir_archivo = os.path.join(root, file)
            
            # Crear estructura equivalente en la carpeta de salida
            relative_path = os.path.relpath(root, dir_origen)
            output_subdir = os.path.join(dir_destino, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Procesar imagen
            img = cv2.imread(dir_archivo)
            if img is not None:

                #normalizar brillo
                img = normalize_brightness(img)

                #Redimensionar
                resized_img = cv2.resize(img, tamanio_objetivo, interpolation=cv2.INTER_AREA)
                output_path = os.path.join(output_subdir, file)
                cv2.imwrite(output_path, resized_img)
            else:
                print(f"Error al cargar {dir_archivo}")
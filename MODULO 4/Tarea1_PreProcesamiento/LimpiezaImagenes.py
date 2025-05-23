#pip install opencv-python 
import cv2
import os
import shutil
import numpy as np

# Directorios
#La de menor resolución es disgust286 con 13x17
dir_origen = "C:/Users/usuario1/OneDrive/Escritorio/IA/fane_data"  # Carpeta raíz con subcarpetas
dir_destino = "C:/Users/usuario1/OneDrive/Escritorio/IA/fane_redimensionado"
dir_fallidas = "C:/Users/usuario1/OneDrive/Escritorio/IA/imagenes_fallidas"

os.makedirs(dir_destino, exist_ok=True)
os.makedirs(dir_fallidas, exist_ok=True)

# Tamaño objetivo (ancho, alto)
tamanio_objetivo = (224, 224) #Numero aureo? 16:10 aprox

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
            imgOriginal = cv2.imread(dir_archivo)

            # Verificar si la imagen se ha cargado correctamente
            if imgOriginal is None:
                # Mover la imagen a la carpeta de fallidas
                shutil.move(dir_archivo, os.path.join(dir_fallidas, file))
                print(f"Imagen fallida: {dir_archivo}")
                continue
            
            # Obtener dimensiones de la imagen original
            height, width, channels = imgOriginal.shape
            
            #normalizar brillo
            img = normalize_brightness(imgOriginal)

            #Redimensionar
            img = cv2.resize(img, tamanio_objetivo, interpolation=cv2.INTER_CUBIC)

            #Borrosear
            if(height<75 and width<75):
                img = cv2.GaussianBlur(img, (21, 21), sigmaX=0)
            elif(height<150 and width<150):
                img = cv2.GaussianBlur(img, (11, 11), sigmaX=0)
            elif(height<224 and width<224):
                img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

            #Aumentar contraste
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.std() < 30:
                try:
                    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                    img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0])
                    img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
                except Exception as e:
                    print(f"Error aplicando CLAHE en {file}: {e}")


            #Escribir
            output_path = os.path.join(output_subdir, file)
            cv2.imwrite(output_path, img)

        else:
             print(f"Error al cargar {dir_archivo}")
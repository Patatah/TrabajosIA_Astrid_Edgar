#pip install opencv-python 
import cv2
import os
import shutil
import numpy as np

# Directorios
#La de menor resolución es disgust286 con 13x17
#"C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_data"
dir_origen = "C:/Users/usuario1/OneDrive/Escritorio/IA/fane_data"  # Carpeta raíz con subcarpetas
dir_destino = "C:/Users/usuario1/OneDrive/Escritorio/IA/fane_redimensionado"
dir_fallidas = "C:/Users/usuario1/OneDrive/Escritorio/IA/imagenes_fallidas"

os.makedirs(dir_destino, exist_ok=True)
os.makedirs(dir_fallidas, exist_ok=True)

# Tamaño objetivo (ancho, alto)
tamanio_objetivo = (224, 224) #Numero aureo? 16:10 aprox

##func normalizar brillo
def normaliza_brillo(img, target_mean=128):
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


def aumentar_contraste(img, alpha):

    # Convertir a flotante y normalizar a [0, 1] si es necesario
    img_float = img.astype(np.float32) / 255.0
    
    # Aplicar transformación lineal en cada canal (B, G, R)
    img_contrastada = np.clip(alpha * (img_float - 0.5) + 0.5, 0, 1)
    
    # Volver a rango [0, 255] y tipo uint8
    img_contrastada = (img_contrastada * 255).astype(np.uint8)
    
    return img_contrastada

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

            if imgOriginal is None:
                shutil.move(dir_archivo, os.path.join(dir_fallidas, file))
                print(f"Imagen fallida: {dir_archivo}")
                continue
            
            # Obtener dimensiones de la imagen original
            height, width, channels = imgOriginal.shape
            
            #normalizar brillo
            img = normaliza_brillo(imgOriginal)

            #Redimensionar
            img = cv2.resize(img, tamanio_objetivo, interpolation=cv2.INTER_CUBIC)

            #Borrosear
            if(height<50 and width<50):
                img = cv2.GaussianBlur(img, (19, 19), sigmaX=0)
            elif(height<75 and width<75):
                img = cv2.GaussianBlur(img, (17, 17), sigmaX=0)
            elif(height<100 and width<100):
                img = cv2.GaussianBlur(img, (15, 15), sigmaX=0)
            elif(height<125 and width<125):
                img = cv2.GaussianBlur(img, (13, 13), sigmaX=0)
            elif(height<150 and width<150):
                img = cv2.GaussianBlur(img, (9, 9 ), sigmaX=0)
            elif(height<224 and width<224):
                img = cv2.GaussianBlur(img, (5, 5), sigmaX=0)

            #Normalizar contraste
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.std() < 30:
                try:
                    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                    img_ycrcb[:, :, 0] = clahe.apply(img_ycrcb[:, :, 0])
                    img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
                except Exception as e:
                    print(f"Error aplicando CLAHE en {file}: {e}")

            #Aumentar contraste

            img = aumentar_contraste(img, 1.2)


            #Escribir
            output_path = os.path.join(output_subdir, file)
            cv2.imwrite(output_path, img)

        else:
             print(f"Error al cargar {dir_archivo}")
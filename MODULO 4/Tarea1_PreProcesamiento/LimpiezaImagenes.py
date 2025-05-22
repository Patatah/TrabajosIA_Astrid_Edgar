#pip install opencv-python 
import cv2
import os

# Directorios
#La de menor resolución es disgust286 con 13x17
input_dir = "C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_data"  # Carpeta raíz con subcarpetas
output_dir = "C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_redimensionado"
os.makedirs(output_dir, exist_ok=True)

# Tamaño objetivo (ancho, alto)
target_size = (50, 80) #Numero aureo? 16:10 aprox

# Recorrer todas las subcarpetas
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.jpg')):
            input_path = os.path.join(root, file)
            
            # Crear estructura equivalente en la carpeta de salida
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Procesar imagen
            img = cv2.imread(input_path)
            if img is not None:
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                output_path = os.path.join(output_subdir, file)
                cv2.imwrite(output_path, resized_img)
            else:
                print(f"Error al cargar {input_path}")
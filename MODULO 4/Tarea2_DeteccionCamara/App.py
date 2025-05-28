#pip install opencv-python
import cv2

img_ultimo_rostro = None
x,y,w,h = 0,0,0,0 ##posiciones del rectangulo verde
contador_frames = 0
intervalo_deteccion = 3


def detectar_rostros():
    global img_ultimo_rostro
    global x,y,w,h
    global contador_frames
    global intervalo_deteccion

    
    
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
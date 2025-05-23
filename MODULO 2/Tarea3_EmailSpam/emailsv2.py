import re # expresiones regulares
import pandas as pd # para leer el csv
from sklearn.feature_extraction.text import CountVectorizer # sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib # guardar el modelo
import tkinter as tk
from tkinter import messagebox



def existe_modelo():
    return os.path.exists('MODULO 2/Tarea3_EmailSpam/modelo.pkl')

def limpieza_texto(texto):
    texto = texto.lower() # Minúsculas
    texto = texto[9:] # Eliminamos la palabra subject: del texto
    # Uno o mas espacios espacios (/s+), remplazados por uno solo en el texto.
    texto = re.sub(r'\s+', ' ', texto)  #
    # r (raw). Entre corchetes escribimos un conjunto. ^ es negación. \w alfanumerico. \s son espacios.
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto


def subir_texto(): # Procesar si el email es spam
    print("Hola mundo")
    texto = campo_texto.get("1.0", tk.END)
    if texto.strip() == "":
        messagebox.showwarning("Advertencia", "Por favor ingresa un texto.")
        return
    else:
        texto = limpieza_texto(texto)
        texto_vec = vectorizer.transform([texto])
        prediccion = model.predict(texto_vec)[0]
        if prediccion == "spam":
            messagebox.showinfo("Resultado", "El email es Spam.")
        else:
            messagebox.showinfo("Resultado", "El email es Ham.")

# Con pandas podemos leer los datos y luego los guardamos en variables
# spam_ham_dataset.csv
data = pd.read_csv('MODULO 2/Tarea3_EmailSpam/spam_ham_dataset.csv')
textos = data['text'].apply(limpieza_texto)  # Asi se llama la columna en el csv
labels = data['label']  # "Ham" o "Spam" en el csv. 

# Convertimos los textos a numeros que entiende el modelo 
vectorizer = CountVectorizer() 
X = vectorizer.fit_transform(textos) # Este tipo de objeto convierte texto a vectores que cuentan ocurrencias de palabras

# Dividiendo en entrenamiento y prueba
# Train test split es una funcion de scikit que divide el dataset de manera aleatoria, le pasamos los textos ya vectorizados
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2) # 80% entrenamiento, 20% prueba

if not existe_modelo():
    print("No se encontró el modelo, entrenando uno nuevo")
    
    # Modelo naive bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)
    joblib.dump(model, 'MODULO 2/Tarea3_EmailSpam/modelo.pkl') # Guardar el modelo recién entrenado
else: 
    print("Modelo encontrado en los archivos, cargando.")
    model = joblib.load('MODULO 2/Tarea3_EmailSpam/modelo.pkl') # Cargar el modelo ya entrenado
    
print("------------------------------------------------------")
# Esto comprueba que tantas del test salieron bien y nos da un porcentaje de predicción
prediccion = model.predict(X_test)
print(f"Precisión: {accuracy_score(y_test, prediccion)*100:.2f}%")
print  ("------------------------------------------------------")

# Probando con un email
email = ["Subject: Win a lot of money fast exclusive offer just for you"]
print(email)
email = [limpieza_texto(email) for email in email]
email_vec = vectorizer.transform(email)
print(f"¿Es spam? {model.predict(email_vec)[0]}")
print  ("------------------------------------------------------")

# Probando con otro email
email2 = ["Subject: Hey this is Astrid from work, I need your help with something"]
print(email2)
email2 = [limpieza_texto(email) for email in email2]
email2_vec = vectorizer.transform(email2)
print(f"¿Es spam? {model.predict(email2_vec)[0]}")

ventana = tk.Tk()
ventana.title("Analizador de Spam o Ham")
ventana.geometry("400x300")

#Apartado grafico

tk.Label(ventana, text="Ingresa tu email:").pack(pady=5)
campo_texto = tk.Text(ventana, height=10, width=40)
campo_texto.pack(pady=5)

boton_subir = tk.Button(ventana, text="Subir Texto", command=subir_texto)
boton_subir.pack(pady=10)

# Iniciar la ventana
ventana.mainloop()
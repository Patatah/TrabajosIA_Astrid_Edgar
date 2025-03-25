# Recuerda correr "pip install pandas scikit-learn" en la terminal para instalar las librerías necesarias

import pandas as pd # para leer el csv
import re # expresiones regulares
import joblib # guardar el modelo
from sklearn.feature_extraction.text import CountVectorizer # sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def limpieza_texto(texto):
    texto = texto.lower() # Minúsculas
    texto = texto[9:] # Eliminamos la palabra subject: del texto
    # Uno o mas espacios espacios (/s+), remplazados por uno solo en el texto.
    texto = re.sub(r'\s+', ' ', texto)  #
    # r (raw). Entre corchetes escribimos un conjunto. ^ es negación. \w alfanumerico. \s son espacios.
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def existe_modelo():
    return os.path.exists('Modulo 2/Tarea3_EmailSpam/modelo.pkl')

# Con pandas podemos leer los datos y luego los guardamos en variables
# spam_ham_dataset.csv
data = pd.read_csv('Modulo 2/Tarea3_EmailSpam/spam_ham_dataset.csv')
textos = data['text'].apply(limpieza_texto)  # Asi se llama la columna en el csv
labels = data['label']  # "Ham" o "Spam" en el csv. 

# Convertimos los textos a numeros que entiende el modelo 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# Dividiendo en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2) # 80% entrenamiento, 20% prueba


if not existe_modelo():
    print("No se encontró el modelo, entrenando uno nuevo")
    
    # Modelo naive bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)
    joblib.dump(model, 'Modulo 2/Tarea3_EmailSpam/modelo.pkl') # Guardar el modelo recién entrenado
else: 
    print("Modelo encontrado en los archivos, cargando.")
    model = joblib.load('Modulo 2/Tarea3_EmailSpam/modelo.pkl') # Cargar el modelo ya entrenado

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
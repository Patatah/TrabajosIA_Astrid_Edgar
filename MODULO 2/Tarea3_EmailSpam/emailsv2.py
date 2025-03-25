import re # expresiones regulares
import pandas as pd # para leer el csv
from sklearn.feature_extraction.text import CountVectorizer # sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def limpieza_texto(texto):
    texto = texto.lower() # Minúsculas
    texto = texto[9:] # Eliminamos la palabra subject: del texto
    # Uno o mas espacios espacios (/s+), remplazados por uno solo en el texto.
    texto = re.sub(r'\s+', ' ', texto)  #
    # r (raw). Entre corchetes escribimos un conjunto. ^ es negación. \w alfanumerico. \s son espacios.
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

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

# Naive bayes
model = MultinomialNB()
model.fit(X_train, y_train)
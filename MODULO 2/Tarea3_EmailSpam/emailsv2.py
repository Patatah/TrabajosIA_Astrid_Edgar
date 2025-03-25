import re # expresiones regulares
import pandas as pd # para leer el csv

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
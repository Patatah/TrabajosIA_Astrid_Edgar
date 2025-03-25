import re # expresiones regulares

def limpieza_texto(texto):
    texto = texto.lower() # Minúsculas
    texto = texto[9:] # Eliminamos la palabra subject: del texto
    # Uno o mas espacios espacios (/s+), remplazados por uno solo en el texto.
    texto = re.sub(r'\s+', ' ', texto)  #
    # r (raw). Entre corchetes escribimos un conjunto. ^ es negación. \w alfanumerico. \s son espacios.
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

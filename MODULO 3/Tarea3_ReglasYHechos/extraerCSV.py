#pip install translate 
#pip install pandas

from translate import Translator #Libreria para traducir texto
translator = Translator(from_lang="en", to_lang="es")   

contador = 0
cacheIngles = {}
cacheTraducidos = {}

def traducir(ingrediente):
    #Traducir el ingrediente al español
    global contador
    global cacheIngles
    global cacheTraducidos

    for i in range(len(cacheIngles)):
        if cacheIngles[i] == ingrediente:
            return cacheTraducidos[i]

    traduccion = translator.translate(ingrediente)
    cacheTraducidos[contador] = traduccion
    cacheIngles[contador] = ingrediente
    contador += 1
    return traduccion

import pandas as pd #Para leer el CSV
import re #Expresiones regulares para limpieza de datos


rutaCSV = 'MODULO 3\\Tarea3_ReglasYHechos\\RAW_recipes_filtrado.csv'
df = pd.read_csv(rutaCSV, sep=',', encoding='utf-8')

rutaReglas = 'MODULO 3\\Tarea3_ReglasYHechos\\reglas.txt'
with open(rutaReglas, "w") as archivoSobreEscribir: #Borrar el contenido del archivo antes de escribir
        archivoSobreEscribir.write("")

i=0
while True:
    #Obtener los ingredientes de la receta i
    ingredientes = df['ingredients'].iloc[i] 
    ingredientes = ingredientes.replace('[','').replace(']','')

    #Obtener el nombre de la receta i
    receta = df['name'].iloc[i]
    if isinstance(receta, float):  ##Algunas recetas no tienen nombre y las saltamos completamente
        i+=1
        continue
    receta = receta.replace('.','')
    receta = re.sub(r'\s+', '_', receta) #Uno o mas espacios en blanco
    
 

    #Con esta expresión regular podemos encontrar cadenas entre comillas simples 'cadena'
    #y eliminar las comas que están dentro de esas cadenas usando funciones lambda
    #la función lambda m es la que se encarga de eliminar las comas dentro de las cadenas encontradas
    #(porque los datos vienen con errores)
    ingredientes = re.sub(r"'[^']*'", lambda m: m.group(0).replace(',', ''), ingredientes)
    listaIngredientes = ingredientes.split(',')


    for j in range(len(listaIngredientes)):
        with open(rutaReglas, "a", encoding='utf-8') as archivoAppend:
             #Quitar los espacios al inicio y al final de cada ingrediente
             #Corregir el error con los jugos de fruta que vienen escritos como "orange, juice of"
            ingrediente = listaIngredientes[j].strip()
            ingrediente = ingrediente.replace('juice of','juice')
            ingrediente = ingrediente.replace('.','')
            

            ingrediente = traducir(ingrediente) 
            ingrediente = re.sub(r'\s+', '_', ingrediente) #Uno o mas espacios en blanco


            archivoAppend.write("T("+ ingrediente +")") #Agregarle T() a cada ingrediente
            if(j+1) != len(listaIngredientes): archivoAppend.write(" ∧ ") #Agregarle ∧ entre ingredientes

    #Ahora agregar la receta
    with open(rutaReglas, "a", encoding='utf-8') as archivoAppend:
        archivoAppend.write(" → PuedeHacer("+receta+")\n") #Agregarle la receta al final de la regla

    
    if receta == '': break #salir si ya leiste todos
    i+=1


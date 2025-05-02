import csv
import pandas as pd

columnas_necesarias = ['name', 'ingredients', 'n_ingredients']
df = pd.read_csv('MODULO 3\\Tarea3_ReglasYHechos\\RAW_recipes.csv', usecols=columnas_necesarias)

i=0
while True:
    ingredientes = df['ingredients'].iloc[i] 
    
    print(ingredientes)
    ingredientes = ingredientes.replace('[','').replace(']','').replace(' ', '_').replace('\'','')
    print(ingredientes)
    listaIngredientes = ingredientes.split(',')
    


    receta = df['name'].iloc[i]
    receta = receta.replace(' ','_')

    
    if i==10: break
    if receta == '': break #salir si ya leiste todos
    i=i+1

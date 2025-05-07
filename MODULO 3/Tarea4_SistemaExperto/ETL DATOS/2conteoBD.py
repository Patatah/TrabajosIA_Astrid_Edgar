#python -m pip install pyodbc
#pip install pandas
import pyodbc
from collections import Counter

# Configuración de la conexión
server = 'localhost'  # dirección de el servidor
database = 'IA'
username = 'edgar' 
password = '123' 

#Archivo para guardar ingredientes unicos:
rutaArchivo = 'MODULO 3\\Tarea4_SistemaExperto\\ETL DATOS\\conteo.txt'

#Connection String para SQL Server
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    # Establecer conexión
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    diccionario_conteo = Counter()

    query = "SELECT ingredientes FROM Recetas"
    resultado = cursor.execute(query)
    for fila in resultado:
        listaIngredientes = fila[0].split(',')
        diccionario_conteo.update(listaIngredientes)
    
    with open(rutaArchivo, "w", encoding="utf-8") as archivo:
            for nombre, conteo in diccionario_conteo.items():
                archivo.write(f"{nombre} : {conteo}\n") 
    
   
except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
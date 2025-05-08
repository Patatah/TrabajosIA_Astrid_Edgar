#python -m pip install pyodbc
#pip install pandas
import pyodbc
import pandas as pd #Para leer el CSV
import re #Expresiones regulares para limpieza de datos

# Configuración de la conexión
server = 'localhost'  # dirección de el servidor
database = 'CooklyDB'
username = 'edgar' 
password = '123' 

# Ruta del archivo CSV
rutaCSV = 'MODULO 3\\Tarea4_SistemaExperto\\RAW_recipes.csv'

#Connection String para SQL Server
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    # Establecer conexión
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    query = "INSERT INTO Recetas (idReceta, nombre, minutos, fecha, n_pasos, pasos, n_ingredientes, descripcion) VALUES "
    
    datos = pd.read_csv(rutaCSV, sep=',', encoding='utf-8')
    for columna, fila in datos.iterrows():
        id=fila['id']

        #Obtener el nombre de la receta y limpiar
        nombre=fila['name']
        if isinstance(nombre, float):  ##Algunas recetas no tienen nombre y las saltamos completamente
            continue
        nombre = nombre.replace('.','')
        nombre = nombre.replace("'", "''")
        nombre = nombre.strip() 
        nombre = re.sub(r'\s+', ' ', nombre) #Uno o mas espacios en blanco

        minutos=fila['minutes']
        fecha=fila['submitted']
        n_pasos=fila['n_steps']

        #Obtener los pasos de la receta y limpiar
        pasos=fila['steps']
        pasos = pasos.replace('[','').replace(']','')
        pasos = re.sub(r"'[^']*'", lambda m: m.group(0).replace(',', ''), pasos)
        listaPasos = pasos.split(',')
        for i in range(len(listaPasos)):
            listaPasos[i] = listaPasos[i].strip()
            listaPasos[i] = listaPasos[i].strip("'")
            listaPasos[i] = listaPasos[i].replace("'", "''")
            listaPasos[i] = re.sub(r'\s+', ' ', listaPasos[i])
        pasos = ','.join(listaPasos)

        n_ingredientes=fila['n_ingredients']

        descripcion=fila['description']
        if isinstance(descripcion, float):  ##Algunas recetas no tienen descripción
            descripcion = ''
        descripcion = descripcion.replace("'", "''")
        descripcion = descripcion.replace("\r", " ")
        descripcion = descripcion.replace("\n", " ")
        descripcion = re.sub(r'\s+', ' ', descripcion) #Uno o mas espacios en blanco

        #Ejectutar el query
        queryFinal = query+"("+str(id)+", '"+nombre+"', "+str(minutos)+", '"+fecha+"', "+str(n_pasos)+", '"+pasos+"', "+str(n_ingredientes)+", '"+descripcion+"')"
        ##print (queryFinal)
        cursor.execute(queryFinal)
        conn.commit()
        ##print(f'Insertado: {id} - {nombre}')
        
   


    # Ejecutar un insert
    #cursor.execute('')
    #conn.commit()
   
except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
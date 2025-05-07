#python -m pip install pyodbc
import pyodbc

def limpiarIngredientes(lista):
    """
    Función para limpiar los ingredientes de una receta.
    Convierte ingredientes especificos a generales, usando muchas reglas.
    Retorna un string con los ingredientes separados por comas.
    """
    for i in range(len(lista)):
        ingrediente = lista[i]
        ingrediente = ingrediente.replace("'", "''")
        if "sugar" in ingrediente:
            lista[i] = 'sugar'
            continue
        elif "chicken" in ingrediente:
            lista[i] = 'chicken'
            continue

    lista.sort()
    listaLimpia = ",".join(lista)
    return listaLimpia

# Configuración de la conexión
server = 'localhost'  # dirección de el servidor
database = 'IA'
username = 'edgar' 
password = '123' 

#Connection String para SQL Server
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    # Establecer conexión
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursorUpdate = conn.cursor()

    query = "SELECT id, ingredientes FROM Recetas"
    cursor.execute(query)
    resultado = cursor.fetchall()
    for fila in resultado:
        listaIngredientes = fila[1].split(',')
        ingredientesLimpios = limpiarIngredientes(listaIngredientes)
        query = "UPDATE Recetas SET ingredientes = '"+ingredientesLimpios+"' "+"WHERE id = "+str(fila[0])
        print(query)
        cursorUpdate.execute(query)
        conn.commit()
   
except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
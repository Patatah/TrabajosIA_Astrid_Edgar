#python -m pip install pyodbc
import pyodbc

def limpiarIngredientes(lista):
    """
    Función para limpiar los ingredientes de una receta.
    Convierte ingredientes especificos a generales, usando muchas reglas.
    Retorna un string con la query a ejecutar para actualizar la base de datos.
    """
    listaOriginal = lista.copy()
    
    for i in range(len(lista)):
        ingrediente = lista[i]
        if "sugar" in ingrediente or "":
            if "free" in ingrediente:
                continue
            if "brown" not in ingrediente:
                lista[i] = 'sugar'
            else:
                lista[i] = 'brown sugar'
            continue
        elif "sugar-free" in ingrediente:
            lista[i] = lista[i].replace("sugar-free", "").strip()
            continue
        elif "fat-free" in ingrediente:
            lista[i] = lista[i].replace("fat-free", "").strip()
            continue
        elif "sugar free" in ingrediente:
            lista[i] = lista[i].replace("sugar free", "").strip()
            continue
        elif "fat free" in ingrediente:
            lista[i] = lista[i].replace("fat free", "").strip()
            continue
        elif "nonfat" in ingrediente:
            lista[i] = lista[i].replace("nonfat", "").strip()
            continue
        elif "non fat" in ingrediente:
            lista[i] = lista[i].replace("non fat", "").strip()
            continue
        elif "non-fat" in ingrediente:
            lista[i] = lista[i].replace("non-fat", "").strip()
            continue
        elif "substitute" in ingrediente:
            lista[i] = lista[i].replace("substitute", "").strip()
            continue
        elif "margerine" in ingrediente:
            lista[i] = "margarine"
            continue
        elif "beef broth" in ingrediente:
            lista[i] = "beef broth"
            continue
        elif "chicken broth" in ingrediente:
            lista[i] = "beef broth"
            continue

    lista.sort()

    if listaOriginal == lista:
        return None

    for i in range(len(lista)):
        lista[i] = lista[i].replace("'", "''") # Evitar problemas con comillas simples

    
    stringLista = ",".join(lista)
    query = "UPDATE Recetas SET ingredientes = '"+stringLista+"' "+"WHERE id = "+str(fila[0])
    return query

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

        query = limpiarIngredientes(listaIngredientes)
        if query is None:
            continue
        print(query)
        cursorUpdate.execute(query)
        conn.commit()
   
except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
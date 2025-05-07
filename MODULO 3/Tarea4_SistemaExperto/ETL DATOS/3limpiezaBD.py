#python -m pip install pyodbc
import re
import pyodbc

def limpiarIngredientes(lista):
    """
    Función para limpiar los ingredientes de una receta.
    Convierte ingredientes especificos a generales, usando muchas reglas.
    Retorna un string con la query a ejecutar para actualizar la base de datos.
    """
    if lista == None:
        return None
        
    listaOriginal = lista.copy()

    removidos = 0
    for i in range(len(lista)):

        if(i+removidos) >= len(lista): #Manejar cambios de length cuando removemos ingredientes
            break

        lista[i]=re.sub(r'\s+', ' ', lista[i])
        ingrediente = lista[i]
        if "sugar" in ingrediente:
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
        elif "eggs" in ingrediente:
            if("chocolate" in ingrediente or "quail" in ingrediente or "powdered" in ingrediente or "pickled" in ingrediente):
                continue
            lista[i] = "eggs"
            continue
        elif "chocolate" in ingrediente and "egg" in ingrediente:
            lista[i] = "chocolate eggs"
            continue
        elif ingrediente == "egg":
            lista[i] = "eggs"
            continue
        elif "lemon juice" in ingrediente:
            lista[i] = "lemon juice"
            continue
        elif "orange juice" in ingrediente:
            lista[i] = "orange juice"
            continue
        elif "pineapple juice" in ingrediente:
            if "teriyaki" in ingrediente:
                continue
            lista[i] = "pineapple juice"
            continue
        elif "apple juice" in ingrediente:
            if "pineapple" in ingrediente:
                continue
            lista[i] = "apple juice"
            continue
        elif "all-purpose flour" in ingrediente:
            lista[i] = "flour"
            continue
        elif "kosher salt" in ingrediente:
            lista[i] = "salt"
            continue
        elif "sea salt" in ingrediente:
            lista[i] = "salt"
            continue
        elif "garlic clove" in ingrediente:
            lista[i] = "garlic"
            continue
        elif "fresh garlic" in ingrediente:
            lista[i] = "garlic"
            continue
        elif "fresh" in ingrediente:
            lista[i] = lista[i].replace("fresh", "").strip()
            continue
        elif "baking powder" in ingrediente:
            lista[i] = "baking powder"
            continue
        elif "vegetable oil" in ingrediente:
            lista[i] = "vegetable oil"
            continue
        elif "worcestershire sauce" in ingrediente:
            lista[i] = "worcestershire sauce"
            continue
        elif "black pepper" in ingrediente:
            lista[i] = "black pepper"
            continue
        elif "sour cream" in ingrediente:
            if "potato chips" in ingrediente:
                continue
            lista[i] = "sour cream"
            continue
        elif "simply" in ingrediente:
            lista[i] = lista[i].replace("simply", "").strip()
            continue
        elif "extra virgin olive oil" in ingrediente:
            lista[i] = "olive oil"
            continue
        elif "dijon mustard" in ingrediente:
            lista[i] = "dijon mustard"
            continue
        elif "cheddar cheese" in ingrediente:
            lista[i] = "cheddar cheese"
            continue
        elif "campbell's" in ingrediente:
            lista[i] = lista[i].replace("campbell's", "").strip()
            continue
        elif "cream cheese" in ingrediente:
            lista[i] = "cream cheese"
            continue
        elif "cayenne pepper" in ingrediente:
            lista[i] = "cayenne pepper"
            continue
        elif "ground cumin" in ingrediente:
            lista[i] = "cumin"
            continue
        elif "ground beef" in ingrediente:
            lista[i] = "ground beef"
            continue
        elif "ground turkey" in ingrediente:
            lista[i] = "ground turkey"
            continue
        elif "ground chicken" in ingrediente:
            lista[i] = "ground chicken"
        elif "mozzarella cheese" in ingrediente:
            lista[i] = "mozzarella cheese"
            continue
        elif "parmesan cheese" in ingrediente:
            lista[i] = "parmesan cheese"
            continue
        elif "heavy cream" in ingrediente:
            lista[i] = "heavy cream"
            continue
        elif "chili powder" in ingrediente:
            lista[i] = "chili powder"
            continue
        elif "dried oregano" in ingrediente:
            lista[i] = "oregano"
            continue
        elif "dried basil" in ingrediente:
            lista[i] = "basil"
            continue
        elif "dried thyme" in ingrediente:
            lista[i] = "thyme"
            continue
        elif "dried parsley" in ingrediente:
            lista[i] = "parsley"
            continue
        elif "dried rosemary" in ingrediente:
            lista[i] = "rosemary"
            continue
        elif "dried dill" in ingrediente:
            lista[i] = "dill"
            continue
        elif "juice of" == ingrediente:
            lista.remove(ingrediente)
            removidos +=1
            continue
        elif "chicken breast" in ingrediente:
            lista[i] = "chicken breast"
            continue
        elif "chicken thighs" in ingrediente:
            lista[i] = "chicken thighs"
            continue
        elif "chicken legs" in ingrediente:
            lista[i] = "chicken legs"
            continue
        elif "chicken wings" in ingrediente:
            lista[i] = "chicken wings"
            continue
        elif "ketchup" in ingrediente:
            lista[i] = "ketchup"
            continue
        elif "balsamic vinegar" in ingrediente:
            lista[i] = "balsamic vinegar"
            continue
        elif "ground ginger" in ingrediente:
            lista[i] = "ginger"
            continue
        elif "diced tomatoes" in ingrediente:
            lista[i] = "diced tomatoes"
            continue
        elif "tomato sauce" in ingrediente:
            lista[i] = "tomato sauce"
            continue
        elif "tomato paste" in ingrediente:
            lista[i] = "tomato paste"
            continue
        elif "chili sauce" in ingrediente:
            lista[i] = "chili sauce"
            continue
        elif "soy sauce" in ingrediente:
            lista[i] = "soy sauce"
            continue
        elif "rice vinegar" in ingrediente:
            lista[i] = "rice vinegar"
            continue
        elif "white vinegar" in ingrediente:
            lista[i] = "white vinegar"
            continue
        elif "red wine vinegar" in ingrediente:
            lista[i] = "red wine vinegar"
            continue
        elif "apple cider vinegar" in ingrediente:
            lista[i] = "apple cider vinegar"
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
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
                lista[i] = ingrediente.replace("sugar free", "").strip()
                lista[i] = ingrediente.replace("sugar-free", "").strip()
            if "brown" not in ingrediente:
                lista[i] = 'sugar'
            else:
                lista[i] = 'brown sugar'
        elif "fat-free" in ingrediente:
            lista[i] = lista[i].replace("fat-free", "").strip()
        elif "fat free" in ingrediente:
            lista[i] = lista[i].replace("fat free", "").strip()
        elif "nonfat" in ingrediente:
            lista[i] = lista[i].replace("nonfat", "").strip()
        elif "non fat" in ingrediente:
            lista[i] = lista[i].replace("non fat", "").strip()
        elif "non-fat" in ingrediente:
            lista[i] = lista[i].replace("non-fat", "").strip()
        elif "substitute" in ingrediente:
            lista[i] = lista[i].replace("substitute", "").strip()
        elif "margerine" in ingrediente:
            lista[i] = "margarine"
        elif "beef broth" in ingrediente:
            lista[i] = "beef broth"
        elif "chicken broth" in ingrediente:
            lista[i] = "beef broth"
        elif "eggs" in ingrediente:
            if not("chocolate" in ingrediente or "quail" in ingrediente or "powdered" in ingrediente or "pickled" in ingrediente):
                lista[i] = "eggs"
        elif "chocolate" in ingrediente and "egg" in ingrediente:
            lista[i] = "chocolate eggs"
        elif ingrediente == "egg":
            lista[i] = "eggs"
        elif "lemon juice" in ingrediente:
            lista[i] = "lemon juice"
        elif "orange juice" in ingrediente:
            lista[i] = "orange juice"
        elif "pineapple juice" in ingrediente:
            if "teriyaki" not in ingrediente:
                lista[i] = "pineapple juice"
        elif "apple juice" in ingrediente:
            if "pineapple" not in ingrediente:
                lista[i] = "apple juice"
        elif "all-purpose flour" in ingrediente:
            lista[i] = "flour"
        elif "kosher salt" in ingrediente:
            lista[i] = "salt"
        elif "sea salt" in ingrediente:
            lista[i] = "salt"
        elif "garlic clove" in ingrediente:
            lista[i] = "garlic"
        elif "fresh garlic" in ingrediente:
            lista[i] = "garlic"
        elif "fresh" in ingrediente:
            lista[i] = lista[i].replace("fresh", "").strip()
        elif "baking powder" in ingrediente:
            lista[i] = "baking powder"
        elif "vegetable oil" in ingrediente:
            lista[i] = "vegetable oil"
        elif "worcestershire sauce" in ingrediente:
            lista[i] = "worcestershire sauce"
        elif "black pepper" in ingrediente:
            lista[i] = "black pepper"
        elif "sour cream" in ingrediente:
            if "potato chips" in ingrediente:
                continue
            lista[i] = "sour cream"
        elif "simply" in ingrediente:
            lista[i] = lista[i].replace("simply", "").strip()
        elif "extra virgin olive oil" in ingrediente:
            lista[i] = "olive oil"
        elif "dijon mustard" in ingrediente:
            lista[i] = "dijon mustard"
        elif "cheddar cheese" in ingrediente:
            lista[i] = "cheddar cheese"
        elif "campbell's" in ingrediente:
            lista[i] = lista[i].replace("campbell's", "").strip()
        elif "cream cheese" in ingrediente:
            lista[i] = "cream cheese"
        elif "cayenne pepper" in ingrediente:
            lista[i] = "cayenne pepper"
        elif "ground cumin" in ingrediente:
            lista[i] = "cumin"
        elif "ground beef" in ingrediente:
            lista[i] = "ground beef"
        elif "ground turkey" in ingrediente:
            lista[i] = "ground turkey"
        elif "ground chicken" in ingrediente:
            lista[i] = "ground chicken"
        elif "mozzarella cheese" in ingrediente:
            lista[i] = "mozzarella cheese"
        elif "parmesan cheese" in ingrediente:
            lista[i] = "parmesan cheese"
        elif "heavy cream" in ingrediente:
            lista[i] = "heavy cream"
        elif "chili powder" in ingrediente:
            lista[i] = "chili powder"
        elif "dried oregano" in ingrediente:
            lista[i] = "oregano"
        elif "dried basil" in ingrediente:
            lista[i] = "basil"
        elif "dried thyme" in ingrediente:
            lista[i] = "thyme"
        elif "dried parsley" in ingrediente:
            lista[i] = "parsley"
        elif "dried rosemary" in ingrediente:
            lista[i] = "rosemary"
        elif "dried dill" in ingrediente:
            lista[i] = "dill"
        elif "juice of" == ingrediente:
            lista.remove(ingrediente)
            removidos +=1
        elif "chicken breast" in ingrediente:
            lista[i] = "chicken breast"
            continue
        elif "chicken thigh" in ingrediente:
            lista[i] = "chicken thighs"
            continue
        elif "chicken leg" in ingrediente:
            lista[i] = "chicken legs"
            continue
        elif "chicken wing" in ingrediente:
            lista[i] = "chicken wings"
        elif "ketchup" in ingrediente:
            lista[i] = "ketchup"
        elif "balsamic vinegar" in ingrediente:
            lista[i] = "balsamic vinegar"
        elif "ground ginger" in ingrediente:
            lista[i] = "ginger"
        elif "diced tomatoes" in ingrediente:
            lista[i] = "diced tomatoes"
            continue
        elif "tomato sauce" in ingrediente:
            lista[i] = "tomato sauce"
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
        elif "low-fat" in ingrediente:
            lista[i] = lista[i].replace("low-fat", "").strip()
            


    lista.sort()

    if listaOriginal == lista:
        return None

    for i in range(len(lista)):
        lista[i] = lista[i].replace("'", "''") # Evitar problemas con comillas simples

    
    stringLista = ",".join(lista)
    
    return stringLista

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
    resultado=cursor.execute(query)
    resultado = cursor.fetchall()
    for fila in resultado:
        listaIngredientes = fila[1].split(',')
        print(fila[0], listaIngredientes)
        
        ingredientesActualizados = limpiarIngredientes(listaIngredientes)
        print(ingredientesActualizados)
        if ingredientesActualizados is None:
            continue
        print(query)
        cursorUpdate.execute(query)
        conn.commit()
   
except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
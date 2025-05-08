#python -m pip install pyodbc
#pip install pandas
import pyodbc
from collections import Counter
import pandas as pd #Para leer el CSV
import re #Expresiones regulares para limpieza de datos

def limpiarIngrediente(ingrediente):
    """
    Función para limpiar un ingrediente de una receta.
    Convierte ingredientes especificos a generales, usando muchas reglas.
    Retorna un string con el ingrediente limpio.
    """
    if ingrediente == None:
        return None
    
    ingrediente = ingrediente.strip()
    ingrediente = ingrediente.strip("'")
    ingrediente = ingrediente.strip('"')
    ingrediente = re.sub(r'"+', ' ', ingrediente)
    ingrediente = re.sub(r'\s+', ' ', ingrediente)

    if "sugar" in ingrediente:
        if "free" in ingrediente:
            ingrediente = ingrediente.replace("sugar free", "").strip()
            ingrediente = ingrediente.replace("sugar-free", "").strip()
        elif "brown" not in ingrediente:
                ingrediente = 'sugar'
        else:
                ingrediente = 'brown sugar'
    if "fat-free" in ingrediente:
        ingrediente = ingrediente.replace("fat-free", "").strip()
    if "fat free" in ingrediente:
        ingrediente = ingrediente.replace("fat free", "").strip()
    if "nonfat" in ingrediente:
        ingrediente = ingrediente.replace("nonfat", "").strip()
    if "non fat" in ingrediente:
        ingrediente = ingrediente.replace("non fat", "").strip()
    if "non-fat" in ingrediente:
        ingrediente = ingrediente.replace("non-fat", "").strip()
    if "substitute" in ingrediente:
        ingrediente = ingrediente.replace("substitute", "").strip()
    if "margerine" in ingrediente:
        ingrediente = "margarine"
    if "beef broth" in ingrediente:
        ingrediente = "beef broth"
    if "chicken broth" in ingrediente:
        ingrediente = "beef broth"
    if "eggs" in ingrediente:
        if not("chocolate" in ingrediente or "quail" in ingrediente or "powdered" in ingrediente or "pickled" in ingrediente):
            ingrediente = "eggs"
    if "chocolate" in ingrediente and "egg" in ingrediente:
        ingrediente = "chocolate eggs"
    if ingrediente == "egg":
        ingrediente = "eggs"
    if ingrediente == "egg ":
        ingrediente = "eggs"
    if "lemon juice" in ingrediente:
        ingrediente= "lemon juice"
    if "orange juice" in ingrediente:
        ingrediente= "orange juice"
    if "pineapple juice" in ingrediente:
        if "teriyaki" not in ingrediente:
            ingrediente = "pineapple juice"
    if "apple juice" in ingrediente:
        if "pineapple" not in ingrediente:
            ingrediente = "apple juice"
    if "all-purpose flour" in ingrediente:
        ingrediente = "flour"
    if "kosher salt" in ingrediente:
        ingrediente = "salt"
    if "sea salt" in ingrediente:
        ingrediente = "salt"
    if "garlic clove" in ingrediente:
        ingrediente = "garlic"
    if "fresh garlic" in ingrediente:
        ingrediente = "garlic"
    if "fresh" in ingrediente:
        ingrediente = ingrediente.replace("fresh", "").strip()
    if "baking powder" in ingrediente:
        ingrediente = "baking powder"
    if "vegetable oil" in ingrediente:
        ingrediente = "vegetable oil"
    if "worcestershire sauce" in ingrediente:
        ingrediente = "worcestershire sauce"
    if "black pepper" in ingrediente:
        ingrediente = "black pepper"
    if "sour cream" in ingrediente:
        if "potato chips" not in ingrediente:
            ingrediente = "sour cream"
    if "simply" in ingrediente:
        ingrediente = ingrediente.replace("simply", "").strip()
    if "extra virgin olive oil" in ingrediente:
        ingrediente = "olive oil"
    if "dijon mustard" in ingrediente:
        ingrediente = "dijon mustard"
    if "cheddar cheese" in ingrediente:
        ingrediente = "cheddar cheese"
    if "campbell's" in ingrediente:
        ingrediente = ingrediente.replace("campbell's", "").strip()
    if "cream cheese" in ingrediente:
        ingrediente = "cream cheese"
    if "cayenne pepper" in ingrediente:
        ingrediente = "cayenne pepper"
    if "ground cumin" in ingrediente:
        ingrediente = "cumin"
    if "ground beef" in ingrediente:
        ingrediente = "ground beef"
    if "ground pork" in ingrediente:
        ingrediente = "ground pork"
    if "ground turkey" in ingrediente:
        ingrediente = "ground turkey"
    if "ground chicken" in ingrediente:
        ingrediente = "ground chicken"
    if "mozzarella cheese" in ingrediente:
        ingrediente = "mozzarella cheese"
    if "parmesan cheese" in ingrediente:
        ingrediente = "parmesan cheese"
    if "heavy cream" in ingrediente:
        ingrediente = "heavy cream"
    if "chili powder" in ingrediente:
        ingrediente = "chili powder"
    if "dried oregano" in ingrediente:
        ingrediente = "oregano"
    if "dried basil" in ingrediente:
        ingrediente = "basil"
    if "dried thyme" in ingrediente:
        ingrediente = "thyme"
    if "dried parsley" in ingrediente:
        ingrediente = "parsley"
    if "dried rosemary" in ingrediente:
        ingrediente = "rosemary"
    if "dried dill" in ingrediente:
        ingrediente = "dill"
    if "juice of" == ingrediente:
        return None
    if "chicken breast" in ingrediente:
        ingrediente = "chicken breast"
    if "chicken thigh" in ingrediente:
        ingrediente = "chicken thighs"
    if "chicken leg" in ingrediente:
        ingrediente = "chicken legs"
    if "chicken wing" in ingrediente:
        ingrediente = "chicken wings"
    if "ketchup" in ingrediente:
        ingrediente = "ketchup"
    if "balsamic vinegar" in ingrediente:
        ingrediente = "balsamic vinegar"
    if "ground ginger" in ingrediente:
        ingrediente = "ginger"
    if "diced tomatoes" in ingrediente:
        ingrediente = "diced tomatoes"
    if "tomato sauce" in ingrediente:
        ingrediente = "tomato sauce"
    if "tomato paste" in ingrediente:
        ingrediente = "tomato paste"
    if "chili sauce" in ingrediente:
        ingrediente = "chili sauce"
    if "soy sauce" in ingrediente:
        ingrediente = "soy sauce"
    if "rice vinegar" in ingrediente:
        ingrediente = "rice vinegar"
    if "white vinegar" in ingrediente:
        ingrediente = "white vinegar"
    if "red wine vinegar" in ingrediente:
        ingrediente = "red wine vinegar"
    if "apple cider vinegar" in ingrediente:
        ingrediente = "apple cider vinegar"
    if "low-fat" in ingrediente:
        ingrediente = ingrediente.replace("low-fat", "").strip()
    if "low fat" in ingrediente:
        ingrediente = ingrediente.replace("low fat", "").strip()
    if "small" in ingrediente:
        ingrediente = ingrediente.replace("small", "").strip()
    if "%" in ingrediente:
        for i in range(0, 100):
            ingrediente = ingrediente.replace(str(i)+"%", "").strip()
    if "egg white" == ingrediente:
        ingrediente = "egg whites"
    if "egg yolk" == ingrediente:
        ingrediente = "egg yolks"

    ingrediente = ingrediente.strip()
    ingrediente = ingrediente.strip("'")
    ingrediente = ingrediente.strip('"')
    ingrediente = ingrediente.replace("'", "''")
    ingrediente = re.sub(r'"+', ' ', ingrediente)
    ingrediente = re.sub(r'\s+', ' ', ingrediente)

    return ingrediente

# Configuración de la conexión
server = 'localhost'  # dirección de el servidor
database = 'CooklyDB'
username = 'edgar' 
password = '123' 

# Ruta del archivo CSV
rutaCSV = 'MODULO 3\\Tarea4_SistemaExperto\\RAW_recipes.csv'

#Archivo para guardar ingredientes unicos:
rutaArchivo = 'MODULO 3\\Tarea4_SistemaExperto\\ETL DATOS\\3postLimpieza.txt'

#Connection String para SQL Server
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    # Establecer conexión
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor2 = conn.cursor()

    diccionario_conteo = Counter()

    datos = pd.read_csv(rutaCSV, sep=',', encoding='utf-8', usecols=['ingredients'])
    for columna, fila in datos.iterrows():
        ingredientes = fila['ingredients']
        ingredientes = ingredientes.replace('[','').replace(']','')
        ingredientes = ingredientes.replace(", juice of", "")
        ingredientes = ingredientes.replace(",juice of", "")
        listaIngredientes = ingredientes.split(',')
        for i in range(len(listaIngredientes)):
            ingrediente = listaIngredientes[i].strip()
            listaIngredientes[i]=limpiarIngrediente(ingrediente)

        diccionario_conteo.update(listaIngredientes)

    # Ahora para insertar los ingredientes únicos en la base de datos
    query = "INSERT INTO Ingredientes (idIngrediente, nombre, ocurrencias) VALUES "
    diccionario_ordenado = diccionario_conteo.most_common()
    
    i=0
    for nombre, conteo in diccionario_ordenado:
        queryFinal= query+"("+str(i)+", '"+nombre+"',"+str(conteo)+")"
        cursor.execute(queryFinal)
        conn.commit()
        i+=1

    # Guardar en la tabla intermedia
    datos = pd.read_csv(rutaCSV, sep=',', encoding='utf-8', usecols=['ingredients', 'id'])
    for columna, fila in datos.iterrows():
        ingredientes = fila['ingredients']
        ingredientes = ingredientes.replace('[','').replace(']','')
        ingredientes = ingredientes.replace(", juice of", "")
        ingredientes = ingredientes.replace(",juice of", "")
        listaIngredientes = ingredientes.split(',')
        for i in range(len(listaIngredientes)):
            ingrediente = listaIngredientes[i].strip()
            listaIngredientes[i]=limpiarIngrediente(ingrediente)

        id=fila['id']
        for i in range(len(listaIngredientes)):
            if ingredientes[i] != None:
                consultaIngrediente = cursor2.execute("SELECT TOP 1 idIngrediente FROM Ingredientes WHERE nombre = '"+listaIngredientes[i]+"'")
                if consultaIngrediente:
                    idIngrediente = consultaIngrediente.fetchone()[0]
                else:
                    idIngrediente = None
                queryFinal = "INSERT INTO RecetaIngrediente (idReceta, idIngrediente) VALUES ("+str(id)+", "+str(idIngrediente)+")"
                print(queryFinal)
                cursor.execute(queryFinal)
                conn.commit()

    
   
except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
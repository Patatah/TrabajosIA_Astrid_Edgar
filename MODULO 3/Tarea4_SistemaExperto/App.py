#pip install customtkinter
#python -m pip install pyodbc

import customtkinter as ctk
import pyodbc
from collections import Counter

# Configuración de la conexión
server = 'localhost'  # dirección de el servidor
database = 'CooklyDB'
username = 'edgar' 
password = '123' 

#Connection String para SQL Server
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    # Establecer conexión
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()


    checkboxes = []
    buscando = "" #Esta variable guarda lo ultimo que buscamos
    total_ingredientes = [] #Aqui recolectamos strings con todos los ingredientes
    ingredientes_posibles = []
    recetas_posibles = []
    ingredientes_seleccionados =["salt", "water"]

    cursor.execute("SELECT nombre From INGREDIENTES ORDER BY Ocurrencias DESC")
    consulta_ingredientes = cursor.fetchall()
    for tupla in consulta_ingredientes:
        nombre = tupla[0]
        total_ingredientes.append(nombre)

    ##Ingredientes a mostrar a la vez:
    max_ingredientes = 50

    #### VENTANA
    ctk.set_appearance_mode("dark")  # Modo oscuro
    ctk.set_default_color_theme("blue")  # Tema azul

    # Crear ventana
    alto = 720
    ancho = 1280
    app = ctk.CTk()
    app.title("Sistema experto Cookly")
    app.geometry(str(ancho)+"x"+str(alto))

    # --- Barra de búsqueda ---
    frame_busqueda = ctk.CTkFrame(app, width=ancho/2)
    frame_busqueda.pack(side=ctk.TOP, pady=20, padx=20, anchor="nw")

    entrada_busqueda = ctk.CTkEntry(frame_busqueda,placeholder_text="Buscar...",width=ancho/2, height=alto/25,corner_radius=10)
    entrada_busqueda.pack(side="left", padx=10)

    #Dos columnas
    dos_columnas = ctk.CTkFrame(app)
    dos_columnas.pack(side="top", pady=20, padx=20, fill="both", expand=True)

    dos_columnas.grid_columnconfigure(0, weight=1)  
    dos_columnas.grid_columnconfigure(1, weight=1)  

    #IZQUIERDA
    columna_izquierda = ctk.CTkFrame(dos_columnas)
    columna_izquierda.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    texto_izquierda = ctk.CTkLabel(columna_izquierda, text="Ingredientes", font=("Arial", 15))
    texto_izquierda.pack(pady=10)

    #Checkboxes scroll
    scrollable_izq = ctk.CTkScrollableFrame(columna_izquierda, width=100, height=alto-250)
    scrollable_izq.pack(side="left",pady=10, padx=20, fill="both", expand=True)
    
    #DERECHA
    columna_derecha = ctk.CTkScrollableFrame(dos_columnas)
    columna_derecha.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    
    texto_derecha = ctk.CTkLabel(columna_derecha, text="Las recetas disponibles apareceran aquí", font=("Arial", 15), wraplength=ancho/2.5, justify="left")
    texto_derecha.pack(side="left", pady=10, padx=20)

    # Evento checkbox presionado
    def checkbox_presionado(val, estado):
        global ingredientes_seleccionados
        if estado==1:
            ingredientes_seleccionados.append(val)
        else:
            ingredientes_seleccionados.remove(val)

        filtrar_posibles() #Actualizar cuando quitas algo de los ingredientes_seleccionados

    ## Llenar checkboxes
    def crearCheckbox(texto):
        #Esta función crea una nueva checkbox y la mete al arreglo global
        global checkboxes

        checkbox = ctk.CTkCheckBox(scrollable_izq, text=texto,  command=lambda val=texto: checkbox_presionado(val, checkbox.get()))
        checkbox.pack(pady=5, anchor="w")
        checkboxes.append(checkbox)
        return checkbox

    def buscar_ingredientes(busquedaPrevia):  #Si le pasas una cadena vacía, buscará lo que tienes escrito en la barra
        global buscando #Lo ultimo que buscamos
        global entrada_busqueda

        if len(busquedaPrevia)>=1:
            texto_busqueda = busquedaPrevia
        else:
            texto_busqueda = entrada_busqueda.get().lower() #Obtener el texto de la barra
    
        if texto_busqueda == "" or not texto_busqueda:
            actualizar_checkboxes(ingredientes_posibles)
            buscando=""
            return
        ingredientesBuscados = [ingrediente for ingrediente in ingredientes_posibles if texto_busqueda in ingrediente.lower()]
        buscando=texto_busqueda
        actualizar_checkboxes(ingredientesBuscados)

    def actualizar_checkboxes(ingredientesBuscados):
        #Esta función decide si agregar o quitar checkboxes
        global checkboxes

        if not checkboxes: #Caso: no hay ni una checkbox
            for item in ingredientesBuscados[:max_ingredientes]:
                crearCheckbox(item)
        
        if(len(ingredientesBuscados)<max_ingredientes):
            for checkbox in scrollable_izq.winfo_children(): #Caso: Necesito borrar algunos checkbox que me sobran (encontré menos de 50)
                if(len(scrollable_izq.winfo_children()) <= len(ingredientesBuscados)):
                    break
                checkbox.destroy()
                checkboxes.remove(checkbox)
                

        if(len(ingredientesBuscados)>max_ingredientes):
            for i in range(max_ingredientes): #Caso: Necesito crear algunos checkbox que me faltan (encontré mas de 50)
                if(len(scrollable_izq.winfo_children()) >= max_ingredientes):
                    break
                crearCheckbox("Cargando...")
                

        i=0
        for checkbox in scrollable_izq.winfo_children(): #Hacer siempre: Actualizar los checkboxes que ya están puestos 
            checkbox.destroy()
            checkboxes.remove(checkbox)
            nuevo_checkbox=crearCheckbox(ingredientesBuscados[i])
            if(ingredientesBuscados[i] in ingredientes_seleccionados):
                nuevo_checkbox.select()
            else:
                nuevo_checkbox.deselect()
            i+=1

        #scrollable_izq.update_idletasks() #Fin del metodo actualizar

    def actualizarTextoDerecha():
        global texto_derecha
        global recetas_posibles
        
        separador="------------------------------------------------------------------------------------------------------"
        texto=separador+"\n"
        for tuplas in recetas_posibles:
            texto+="Receta "+str(tuplas[0])+". Subida el " + str(tuplas[3])+".\n"
            texto+=separador+"\n"
            texto+="★ Nombre: "+tuplas[1]+".\n"
            texto+="★ Tiempo de preparación: "+str(tuplas[2])+" minutos.\n"
            texto+="\n"
            
            texto+="☆ PASOS:\n"
            #Separar los pasos
            pasos=tuplas[5].split(",")
            i=1
            for paso in pasos:
                texto+="     "+str(i)+".- "+paso+"     \n"
                i+=1

            texto+="\n☆ DESCRIPCIÓN:\n"
            texto+=tuplas[7]+"\n"

            #Obtener ingredientes de las recetas
            query="SELECT I.nombre AS Ingrediente, I.idIngrediente as ID FROM Recetas R JOIN RecetaIngrediente " \
            "RI ON R.idReceta = RI.idReceta JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente WHERE R.idReceta = "+str(tuplas[0])+";"
            cursor.execute(query)
            tuplas = cursor.fetchall()
            ingredientes_de_esta_receta = []
            for tupla in tuplas:
                ingredientes_de_esta_receta.append(tupla[0])
            texto+="\n☆ INGREDIENTES:\n"
            i=1
            for ingrediente in ingredientes_de_esta_receta:
                texto+="     "+str(i)+".- "+ingrediente+"\n"
                i+=1
            
            texto+="\n"+separador+"\n"

            

        

        texto_derecha.configure(text=texto)

    def filtrar_posibles():
        global total_ingredientes
        global entrada_busqueda
        global ingredientes_posibles
        global recetas_posibles
        global ingredientes_seleccionados
       
        print(ingredientes_seleccionados)

        #Obtener que recetas se pueden hacer
        query="SELECT top 5 * FROM Recetas r WHERE ( SELECT COUNT(DISTINCT i.nombre) " \
        "FROM RecetaIngrediente ri JOIN Ingredientes i ON ri.idIngrediente = i.idIngrediente " \
        "WHERE ri.idReceta = r.idReceta AND i.nombre IN ('"
        query+="','".join(ingredientes_seleccionados)
        query+="') ) = "+str(len(ingredientes_seleccionados))+";"

        cursor.execute(query)
        recetas_posibles=cursor.fetchall()
        actualizarTextoDerecha()

        #Ahora si para filtrar los ingredientes posibles
        if(len(ingredientes_seleccionados)>=1):
            query="SELECT DISTINCT I.nombre, I.Ocurrencias FROM Ingredientes I JOIN RecetaIngrediente "\
            "RI ON I.idIngrediente = RI.idIngrediente WHERE RI.idReceta IN (SELECT R.idReceta FROM Recetas R WHERE "
            
            flag=False
            for ingrediente in ingredientes_seleccionados: #Añadimos las condiciones a la consulta
                if flag: #Tenemos que agregar un and si es la segunda subquery
                    query+="AND "
                subQuery= "EXISTS (SELECT 1 FROM RecetaIngrediente RI JOIN Ingredientes I ON RI.idIngrediente = I.idIngrediente "\
                "WHERE RI.idReceta = R.idReceta AND I.nombre = '"+ingrediente+"') "
                query+=subQuery
                flag=True

            #Agregar order al final
            query+=") ORDER BY I.Ocurrencias DESC, I.nombre ASC;"

            ingredientes_posibles.clear() ##limpiamos para empezar de cero

            #print(query)

            cursor.execute(query)
            consulta_ingredientes = cursor.fetchall()
            for tupla in consulta_ingredientes:
                ingredientes_posibles.append(tupla[0])
        else: #En el caso de que no tengamos ni un ingrediente
            ingredientes_posibles = total_ingredientes

        buscar_ingredientes(buscando) #Actualizar con pasos extra
        ##Fin del metodo
    
    filtrar_posibles()

    

    ##Boton de busqueda porque me dio flojera reorganizar el código
    boton_busqueda = ctk.CTkButton(frame_busqueda,text="Buscar",width=ancho/10,height=alto/25,corner_radius=10, command=buscar_ingredientes)
    boton_busqueda.pack(side="left")
    app.bind("<Return>", lambda event: buscar_ingredientes(""))

    app.mainloop()
    ### VENTANA

except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
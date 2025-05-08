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

    cursor.execute("SELECT nombre From INGREDIENTES")
    consulta_ingredientes = cursor.fetchall()

    total_ingredientes = [] #Aqui recolectamos strings con todos los ingredientes
    for tupla in consulta_ingredientes:
        nombre = tupla[0]
        total_ingredientes.append(nombre)

    ##Ingredientes a mostrar a la vez:
    max_ingredientes = 100

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
    checkboxes = []

    # Evento checkbox presionado
    def checkbox_presionado(val, estado):
        print (val)
        if estado==0:
            print ("desmarcado")
        else:
            print ("marcado")

    ## Llenar checkboxes
    def crearCheckbox(texto):
        checkbox = ctk.CTkCheckBox(scrollable_izq, text=texto,  command=lambda val=texto: checkbox_presionado(val, checkbox.get()))
        checkbox.pack(pady=5, anchor="w")
        checkboxes.append(checkbox)


    def actualizarCheckboxes(ingredientesBuscados):
        if not checkboxes: #Caso: no hay ni una checkbox
            for item in ingredientesBuscados[:max_ingredientes]:
                crearCheckbox(item)
            return
        
        checkboxesPuestos = scrollable_izq.winfo_children()
        if(len(ingredientesBuscados)<max_ingredientes):
            for checkbox in scrollable_izq.winfo_children(): #Caso: Necesito borrar algunos checkbox que me sobran (encontré menos de 50)
                if(len(scrollable_izq.winfo_children()) <= len(ingredientesBuscados)):
                    break
                checkbox.destroy()
                checkboxes.remove(checkbox)
                

        if(len(ingredientesBuscados)>max_ingredientes):
            for i in range(max_ingredientes): #Caso: Necesito crear algunos checkbox que me faltan (encontré mas de 50)
                crearCheckbox("Cargando...")
                if(len(scrollable_izq.winfo_children()) >= max_ingredientes):
                    break

        i=0
        for checkbox in scrollable_izq.winfo_children(): #Hacer siempre: Actualizar los checkboxes que ya están puestos 
            checkbox.configure(text=ingredientesBuscados[i])
            i+=1

        #scrollable_izq.update_idletasks() #Fin del metodo actualizar
        
            

    actualizarCheckboxes(total_ingredientes)

    #DERECHA
    columna_derecha = ctk.CTkScrollableFrame(dos_columnas)
    columna_derecha.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    
    texto_derecha = ctk.CTkLabel(columna_derecha, text="Esta es una receta muy larga\nblablabla", font=("Arial", 15), justify="left")
    texto_derecha.pack(side="left", pady=10, padx=20)

    
    def filtrar_ingredientes():
        texto_busqueda = entrada_busqueda.get().lower()
        if texto_busqueda == "" or not texto_busqueda:
            actualizarCheckboxes(total_ingredientes)
            return
        ingredientesBuscados = [ingrediente for ingrediente in total_ingredientes if texto_busqueda in ingrediente.lower()]
        actualizarCheckboxes(ingredientesBuscados)

    ##Boton de busqueda porque me dio flojera reorganizar el código
    boton_busqueda = ctk.CTkButton(frame_busqueda,text="Buscar",width=ancho/10,height=alto/25,corner_radius=10, command=filtrar_ingredientes)
    boton_busqueda.pack(side="left")
    app.bind("<Return>", lambda event: filtrar_ingredientes())

    app.mainloop()
    ### VENTANA

except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
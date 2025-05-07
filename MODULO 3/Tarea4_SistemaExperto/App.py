#pip install customtkinter
#python -m pip install pyodbc

import customtkinter as ctk
import pyodbc

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
    frame_busqueda = ctk.CTkFrame(app)
    frame_busqueda.pack(pady=20, padx=20, fill="x")

    entrada_busqueda = ctk.CTkEntry(frame_busqueda,placeholder_text="Buscar...",width=ancho/2, height=alto/25,corner_radius=10)
    entrada_busqueda.pack(side="left", padx=10)

    boton_busqueda = ctk.CTkButton(frame_busqueda,text="Buscar",width=ancho/10,height=alto/25,corner_radius=10)
    boton_busqueda.pack(side="left")

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

    items = ["Opción 1", "Opción 2", "Opción 3", "Opción 4"]
    checkboxes = []

    for item in items:
        checkbox = ctk.CTkCheckBox(scrollable_izq, text=item)
        checkbox.pack(pady=5, anchor="w")
        checkboxes.append(checkbox)

    #Derecha
    columna_derecha = ctk.CTkScrollableFrame(dos_columnas)
    columna_derecha.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    
    texto_derecha = ctk.CTkLabel(columna_derecha, text="Esta es una receta muy larga\nblablabla", font=("Arial", 15))
    texto_derecha.pack(pady=10)

    app.mainloop()
    

    ### VENTANA

except pyodbc.Error as e:
    print(f'Error de conexión: {e}')
    
finally:
    if 'conn' in locals():
        conn.close()
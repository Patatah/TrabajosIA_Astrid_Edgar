from ArbolBinario import ArbolBinario
# Inicializar
arbolito = ArbolBinario()
        
# Insertar
arbolito.insertar("Edgar")
arbolito.insertar("Astrid")
arbolito.insertar("Octavio")
arbolito.insertar("Diego")
arbolito.insertar("Brayan")
arbolito.insertar("Chiquete")
arbolito.insertar("Omar")
arbolito.insertar("Ricardo")
arbolito.insertar("Ximena")
arbolito.insertar("Joselyn")
arbolito.insertar("Andres")
        
# Buscar
busqueda = "Octavio"
if arbolito.buscar(busqueda):
    print("Encontrado: \"" + busqueda +"\"")
else:
    print("No se encuentra: \"" + busqueda +"\"")
        
print()
# Imprimir
arbolito.imprimir()
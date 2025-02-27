import numpy as np
import time
from Logica import Logica

objetivo = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
])

inicial = np.array([
    [1, 2, 3],
    [4, 5, 0],
    [7, 8, 6]
])

logica = Logica(objetivo)

# Aqui es donde se ejecuta el algoritmo y donde tomamos tiempo
inicio = time.time()
logica.buscarSolucion(inicial)
fin = time.time()

print("* Tiempo de ejecucion del algoritmo: ", fin - inicio, " segundos.")
print("-----------------------------------------------------------------")
logica.imprimirMejorSolucion()



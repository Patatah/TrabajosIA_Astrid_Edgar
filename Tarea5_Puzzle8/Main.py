import numpy as np
from Logica import Logica

objetivo = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
])

inicial = np.array([
    [0, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
])

logica = Logica(objetivo)
logica.calcularHeuristica(inicial)
logica.buscarSolucion(inicial)



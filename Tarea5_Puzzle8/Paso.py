class Paso:
    def __init__(self, tablero, padre, nMovimiento, heuristica):
        self.tablero = tablero
        self.padre = padre
        self.movimiento = nMovimiento
        self.heuristica = heuristica
        self.costo = nMovimiento + heuristica


    
        
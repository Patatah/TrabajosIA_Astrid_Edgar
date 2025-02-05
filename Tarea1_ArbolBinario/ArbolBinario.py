from Nodo import Nodo
class ArbolBinario:
    def __init__(self):
        self.raiz = None

    def buscar(self, busco):
        return self._buscar_recursivo(self.raiz, busco)

    def _buscar_recursivo(self, raiz, busco):
        if raiz is None:
            return False
    
        valor = raiz.valor
        
        if busco == valor:
            return True
        
        if busco < valor:
            return self._buscar_recursivo(raiz.izquierdo, busco)
        else:
            return self._buscar_recursivo(raiz.derecho, busco)

    def insertar(self, valor):
        self.raiz = self._insertar_recursivo(self.raiz, valor)

    def _insertar_recursivo(self, raiz, valor):
        if raiz is None:
            return Nodo(valor)
        
        if valor < raiz.valor:
            izquierdo_actual = raiz.izquierdo
            raiz.izquierdo = self._insertar_recursivo(izquierdo_actual, valor)
        elif valor > raiz.valor:
            derecho_actual = raiz.derecho
            raiz.derecho = self._insertar_recursivo(derecho_actual, valor)
        
        return raiz

    def imprimir(self):
        print("Imprimiendo arbol")
        print("-----------------")
        self._imprimir_recursivo(self.raiz)

    def _imprimir_recursivo(self, raiz):
        if raiz is not None:
            self._imprimir_recursivo(raiz.izquierdo)
            print(raiz.valor)
            self._imprimir_recursivo(raiz.derecho)


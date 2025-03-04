# -*- coding: utf-8 -*-
import numpy as np
import math
from Paso import Paso
class Logica:

    def __init__(self, objetivo):
        self.objetivo = objetivo
        self.listaAbierta = []
        self.listaCerrada = []
        self.costoMejorSolucion = math.inf
        self.pasosMejorSolucion = math.inf
        self.mejorSolucion = None

    def buscarSolucion(self, inicial):
        if(np.array_equal(inicial, self.objetivo)): # Si ya está ordenado, no hay que hacer nada
            print("El tablero ya esta ordenado.")
            return
        
        # Aplanar los arreglos para ver si tienen lo mismo
        objetivo_flat = self.objetivo.flatten()
        inicial_flat = inicial.flatten()
        objetivo_sorted = np.sort(objetivo_flat)
        inicial_sorted = np.sort(inicial_flat)
        # Revisamos si el usuario no está intentando algo imposible
        if not np.array_equal(objetivo_sorted, inicial_sorted):
            print("El tablero inicial y el final no tienen las mismas fichas\nNo puedo crear fichas de la nada tilin." )
            return

        #Limpiar variables
        self.listaAbierta = []
        self.listaCerrada = []
        self.costoMejorSolucion = math.inf
        self.pasosMejorSolucion = math.inf
        self.mejorSolucion = None

        print("Buscando solucion")
        print("-Objetivo-")
        Logica.imprimirTablero(self.objetivo)

        self.inicial = inicial
        print("-Inicial-")
        Logica.imprimirTablero(self.inicial)
        self.solucionRecursiva(inicial, None, 0)

    def solucionRecursiva(self, tableroActual, ultimoPaso, nMovimiento):
        if(nMovimiento>31): #Segun el dominio del problema, no puede hacer una solución optima con mas de 31 movimientos
            return

        # Creamos el objeto del paso actual
        heuristica = self.calcularHeuristica(tableroActual)
        pasoActual = Paso(tableroActual, ultimoPaso, nMovimiento, heuristica)

        # Poda: no tiene caso buscar por aquí porque es un peso mas grande que el mejor encontrado
        if pasoActual.costo > self.costoMejorSolucion:
            return

        #Anti ciclos
        ultimoPaso = None
        try:
            ultimoPaso = pasoActual.padre.padre
        except:
            pass
        if(ultimoPaso != None):
            if(np.array_equal(ultimoPaso.tablero, tableroActual)): ##Si volvimos a donde estabamos hace 2 movimientos
                return
        try:
            ultimoPaso = pasoActual.padre.padre.padre.padre
        except:
            pass
        if(ultimoPaso != None):
            if(np.array_equal(ultimoPaso.tablero, tableroActual)): ##Si volvimos a donde estabamos hace 4 movimientos
                return
        #Termina anti ciclos

        # Caso base: todo está en su lugar
        if np.array_equal(tableroActual, self.objetivo):
            if(pasoActual.costo < self.costoMejorSolucion):
                self.costoMejorSolucion = pasoActual.costo
                self.pasosMejorSolucion = pasoActual.movimiento
                print("Solucion encontrada con", pasoActual.movimiento, "movimientos")
                print("El algoritmo sigue trabajando...")
                self.mejorSolucion = pasoActual
            return
        
        sigMovimientos = Logica.movimientosPosibles(tableroActual)
        sigMovimientos = sorted(sigMovimientos, key=self.calcularHeuristica)
               
        for movimiento in sigMovimientos:
            #print("Un siguiente movimiento podria ser: ")
            #Logica.imprimirTablero(movimiento)
            self.solucionRecursiva(movimiento, pasoActual, nMovimiento+1) # Llamada recursiva para probar el sig paso mejor
        
        #self.listaAbierta.remove(pasoActual) ## Ya vimos todas las opciones de este paso, la quitamos de la lista abierta
        self.listaCerrada.append(tableroActual)

    def calcularHeuristica(self, tablero):
       # Este metodo te dices cuantos numeros hay diferentes al objetivo
        comparacion = self.objetivo == tablero
        iguales = np.sum(comparacion)
        return 9-iguales
    
    def imprimirMejorSolucion(self):
        if self.mejorSolucion == None:
            print("* No se encontro solucion")
            return
        print("* Paso por paso de la mejor solucion")
        self.imprimirPasosRecursivo(self.mejorSolucion, self.pasosMejorSolucion)
    
    def imprimirPasosRecursivo(self, pasoActual, movimientosMejorSolucion):
        if pasoActual.padre == None:
            print("Iniciamos con el tablero asi:")
            Logica.imprimirTablero(pasoActual.tablero)
            return
        
        self.imprimirPasosRecursivo(pasoActual.padre, movimientosMejorSolucion-1)
        print("Movimiento #", movimientosMejorSolucion, ":", Logica.detectarTipoDeMovimiento(pasoActual))
        Logica.imprimirTablero(pasoActual.tablero)  

    @staticmethod
    def encontrarVacio(tablero):
        for i in range(3):
            for j in range(3):
                if tablero[i, j] == 0:
                    return (i, j)
    
    @staticmethod
    def movimientosPosibles(tablero):
        movimientos = []
        i, j = Logica.encontrarVacio(tablero)
    
        # Si hay movimiento arriba
        if i > 0:
            nuevo_tablero = tablero.copy()
            nuevo_tablero[i, j], nuevo_tablero[i-1, j] = nuevo_tablero[i-1, j], nuevo_tablero[i,j]
            movimientos.append(nuevo_tablero)
        
        # Movimiento abajo
        if i < 2:
            nuevo_tablero = tablero.copy()
            nuevo_tablero[i,j], nuevo_tablero[i+1,j] = nuevo_tablero[i+1,j], nuevo_tablero[i,j]
            movimientos.append(nuevo_tablero)
        
        # Movimiento izquierda
        if j > 0:
            nuevo_tablero = tablero.copy()
            nuevo_tablero[i,j], nuevo_tablero[i,j-1] = nuevo_tablero[i,j-1], nuevo_tablero[i,j]
            movimientos.append(nuevo_tablero)
        
        # Movimiento derecha
        if j < 2:
            nuevo_tablero = tablero.copy()
            nuevo_tablero[i,j], nuevo_tablero[i,j+1] = nuevo_tablero[i,j+1], nuevo_tablero[i,j]
            movimientos.append(nuevo_tablero)

        return movimientos
    
    
    @staticmethod
    def imprimirTablero(tablero):
        for i in range(3):
            print("|", end="")
            for j in range(3):
                if(tablero[i,j] == 0):
                    print(" ", end="")
                else:
                    print(tablero[i,j], end="")
                if(j < 2):
                    print("  ", end="")
            print("|")
        print("")
    
    @staticmethod
    def imprimirTableroFancy(tablero):
        print("╔═══════╗")
        for i in range(3):
            print("║", end="")
            for j in range(3):
                if(tablero[i,j] == 0):
                    print(" ", end="")
                else:
                    print(tablero[i,j], end="")
                if(j < 2):
                    print("  ", end="")
            print("║")
        print("╚═══════╝")
    
    @staticmethod 
    def detectarTipoDeMovimiento(paso):
        padre = paso.padre
        if(padre == None):
            return "Inicio"
        else:
            i, j = Logica.encontrarVacio(paso.tablero)
            i2, j2 = Logica.encontrarVacio(padre.tablero)
            if(i == i2):
                if(j == j2+1):
                    return "Mover "+ str(paso.tablero[i, j2])+ " hacia la izquierda"
                else:
                    return "Mover "+ str(paso.tablero[i, j2])+ " hacia la derecha"
            else:
                if(i == i2+1):
                    return "Mover "+ str(paso.tablero[i2, j])+ " hacia arriba"
                else:
                    return "Mover "+ str(paso.tablero[i2, j])+ " hacia abajo"
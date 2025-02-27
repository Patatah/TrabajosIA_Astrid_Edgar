# -*- coding: utf-8 -*-
import numpy as np
import math
from Paso import Paso
class Logica:

    def __init__(self, objetivo):
        self.objetivo = objetivo
        self.listaAbierta = []
        self.listaCerrada = []
        self.mejorSolucion = math.inf

    def buscarSolucion(self, inicial):
        print("Buscando solucion")
        print("- Objetivo -")
        Logica.imprimirTablero(self.objetivo)

        self.inicial = inicial
        print("- Inicial -")
        Logica.imprimirTablero(self.inicial)
        self.solucionRecursiva(inicial, None, 0)

    def solucionRecursiva(self, tableroActual, ultimoPaso, nMovimiento):
        # Creamos el objeto del paso actual
        heuristica = self.calcularHeuristica(tableroActual)
        pasoActual = Paso(tableroActual, ultimoPaso, nMovimiento, heuristica)

        indiceEncontrado = -1
        for obj in self.listaAbierta: # Aqui buscamos si ya está visitado este tablero
            if np.array_equal(obj.tablero, tableroActual):
                indiceEncontrado = self.listaAbierta.index(obj)
                break

            if(indiceEncontrado != -1): # Si tenemos un mejor camino, lo remplazamos en la lista abierta
                if pasoActual.costo < self.listaAbierta[indiceEncontrado].costo:
                    self.listaAbierta[indiceEncontrado] = pasoActual
            else: # Si es la primera vez que vemos este tablero, lo agregamos
                self.listaAbierta.append(pasoActual)

        # Caso base: todo está en su lugar
        if np.array_equal(tableroActual, self.objetivo):
            if(pasoActual.costo < self.mejorSolucion):
                self.mejorSolucion = pasoActual.costo
                print("Mejor solucion encontrada con peso: ", pasoActual.costo)
                
            return
        
        # Poda: no tiene caso buscar por aquí porque es un peso mas grande que el mejor encontrado
        if pasoActual.costo > self.mejorSolucion:
            return

        sigMovimientos = Logica.movimientosPosibles(tableroActual)
        sigMovimientos = sorted(sigMovimientos, key=self.calcularHeuristica)
        
        for movimiento in sigMovimientos:
            #print("Un siguiente movimiento podria ser: ")
            #Logica.imprimirTablero(movimiento)


            if(ultimoPaso != None):
                if(np.array_equal(movimiento, ultimoPaso.tablero)): #Si estamos repitiendo movimientos:
                    #print("No se puede regresar al paso anterior")
                    continue

            self.solucionRecursiva(movimiento, pasoActual, nMovimiento+1) # Llamada recursiva para probar el sig paso mejor
        
        #self.listaAbierta.remove(pasoActual) ## Ya vimos todas las opciones de este paso, la quitamos de la lista abierta
        self.listaCerrada.append(pasoActual)

    def calcularHeuristica(self, arreglo): 
        # Este metodo te dices cuantos numeros hay diferentes al objetivo
        comparacion = self.objetivo == arreglo
        iguales = np.sum(comparacion)
        return 9-iguales
    
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
                print(tablero[i,j], end="")
                if(j < 2):
                    print("  ", end="")
            print("║")
        print("╚═══════╝")
    
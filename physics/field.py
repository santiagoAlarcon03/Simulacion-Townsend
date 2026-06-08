"""
Módulo de campo eléctrico.

Define una representación simple de un campo eléctrico uniforme
para la simulación de la descarga de Townsend.

Un campo uniforme mantiene:

    E(x,y,z) = constante

en toda la región de simulación.

Este campo será utilizado para calcular las fuerzas sobre
los electrones y actualizar sus velocidades.
"""

import numpy as np


class ElectricField:
    """
    Representa un campo eléctrico uniforme.

    El campo es definido mediante un vector constante:

        E = (Ex, Ey, Ez)

    y tendrá el mismo valor para cualquier posición
    dentro del dominio de simulación.
    """

    def __init__(self, vector):
        """
        Inicializa el campo eléctrico.

        Parámetros
        ----------
        vector : array-like
            Vector de campo eléctrico:

                [Ex, Ey, Ez]

            expresado normalmente en V/m.
        """

        # Convierte el vector recibido a un arreglo NumPy
        # de tipo flotante.
        self.vector = np.asarray(vector, dtype=float)

    def at(self, positions):
        """
        Obtiene el campo eléctrico en las posiciones indicadas.

        Debido a que el campo es uniforme, todas las partículas
        reciben exactamente el mismo vector de campo eléctrico.

        Parámetros
        ----------
        positions : ndarray
            Matriz Nx3 con las posiciones de las partículas.

        Retorna
        -------
        ndarray
            Matriz Nx3 donde cada fila contiene el mismo
            vector de campo eléctrico.
        """

        # Si no existen partículas,
        # retorna una matriz vacía.
        if positions.size == 0:
            return np.zeros((0, 3))

        # Replica el vector de campo para cada partícula.
        return np.tile(
            self.vector,
            (positions.shape[0], 1)
        )
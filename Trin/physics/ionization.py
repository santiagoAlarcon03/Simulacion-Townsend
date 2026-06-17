"""
Módulo de ionización.

Contiene funciones relacionadas con la generación de nuevas
cargas debido a colisiones electrón-molécula.

La función principal determina qué electrones poseen energía
suficiente para ionizar una molécula y además tienen éxito
según una probabilidad de ionización.
"""

import numpy as np


def sample_ionization(
    energies_ev,
    threshold_ev,
    probability,
    rng
):
    """
    Determina qué electrones producen ionización.

    Para que ocurra ionización deben cumplirse dos condiciones:

    1. La energía del electrón debe superar el umbral
       de ionización del gas.

    2. Debe ocurrir un evento aleatorio exitoso según
       la probabilidad de ionización.

    Parámetros
    ----------
    energies_ev : ndarray
        Energía de cada electrón en electronvoltios (eV).

    threshold_ev : float
        Energía mínima necesaria para ionizar una molécula.

    probability : float o ndarray
        Probabilidad de ionización.

        Puede ser:

        - Un valor único para todos los electrones.
        - Un arreglo con una probabilidad distinta
          para cada electrón.

    rng : numpy.random.Generator
        Generador de números aleatorios.

    Retorna
    -------
    ndarray(bool)

        Máscara booleana donde:

        True  -> ocurrió ionización.
        False -> no ocurrió ionización.
    """

    # Si no existen electrones,
    # no puede existir ionización.
    if energies_ev.size == 0:
        return np.zeros((0,), dtype=bool)

    # Verifica qué electrones tienen energía suficiente.
    above = energies_ev >= threshold_ev

    # Número aleatorio entre 0 y 1 para cada electrón.
    roll = rng.random(energies_ev.size)

    # Convierte la probabilidad a un arreglo NumPy.
    probability = np.asarray(
        probability,
        dtype=float
    )

    # Si se recibe una probabilidad escalar,
    # se replica para todos los electrones.
    if probability.shape == ():
        probability = np.full(
            energies_ev.shape,
            probability,
            dtype=float
        )

    # Ionización:
    # energía suficiente
    # Y
    # éxito probabilístico
    return above & (roll < probability)
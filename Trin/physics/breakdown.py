"""
Módulo encargado de verificar si una descarga de Townsend
ha alcanzado la condición de autosostenimiento.

Una descarga autosostenida ocurre cuando los electrones
generados por ionización y los electrones secundarios
emitidos desde el cátodo son suficientes para mantener
la avalancha indefinidamente.

La condición utilizada es:

    γ (e^(αd) - 1) ≥ 1

donde:

    α = Primer coeficiente de Townsend.
    γ = Segundo coeficiente de Townsend.
    d = Distancia entre electrodos.

Si la condición se cumple, la descarga puede mantenerse
sin necesidad de electrones externos.
"""

import math

from physics import avalanche


def is_self_sustained(config):
    """
    Determina si la descarga es autosostenida.

    Parámetros
    ----------
    config : object
        Objeto de configuración que contiene:

        townsend_A : float
            Constante A del gas.

        townsend_B : float
            Constante B del gas.

        electric_field : float
            Campo eléctrico aplicado (V/m).

        gas_pressure : float
            Presión del gas.

        gap_distance : float
            Distancia entre electrodos.

        secondary_emission_gamma : float
            Coeficiente secundario de Townsend (γ).

    Retorna
    -------
    bool
        True  -> La descarga es autosostenida.
        False -> La descarga no es autosostenida.
    """

    # -------------------------------------------------------
    # Calcular el primer coeficiente de Townsend (α)
    # -------------------------------------------------------
    alpha = avalanche.townsend_alpha(
        config.townsend_A,
        config.townsend_B,
        config.electric_field,
        config.gas_pressure
    )

    # Si no existe ionización efectiva,
    # no puede existir una descarga autosostenida.
    if alpha <= 0.0:
        return False

    # -------------------------------------------------------
    # Calcular el factor de multiplicación
    # e^(αd) - 1
    # -------------------------------------------------------
    multiplier = math.exp(
        alpha * config.gap_distance
    ) - 1.0

    # -------------------------------------------------------
    # Evaluar la condición de Townsend
    # γ(e^(αd)-1) ≥ 1
    # -------------------------------------------------------
    return (
        config.secondary_emission_gamma
        * multiplier
        >= 1.0
    )
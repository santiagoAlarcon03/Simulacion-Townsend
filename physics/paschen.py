"""
Módulo de Ley de Paschen.

La Ley de Paschen permite calcular el voltaje de ruptura
de un gas en función de:

    - Presión del gas.
    - Distancia entre electrodos.
    - Constantes de Townsend del gas.
    - Coeficiente secundario de Townsend.

El voltaje de ruptura corresponde al punto en el cual
la descarga de Townsend se vuelve autosostenida y el
gas pasa de ser aislante a conductor.
"""

import math


def paschen_voltage(
    pressure,
    distance,
    townsend_a,
    townsend_b,
    gamma=0.02
):
    """
    Calcula el voltaje de ruptura usando la Ley de Paschen.

    Parámetros
    ----------
    pressure : float
        Presión del gas.

    distance : float
        Distancia entre electrodos.

    townsend_a : float
        Constante A de Townsend para el gas.

    townsend_b : float
        Constante B de Townsend para el gas.

    gamma : float, opcional
        Segundo coeficiente de Townsend.
        Valor típico: 0.01 - 0.05

    Retorna
    -------
    float

        Voltaje de ruptura.

        Si el cálculo no es físicamente válido,
        retorna infinito.
    """

    # Producto presión-distancia.
    pd = pressure * distance

    # No existe descarga física.
    if pd <= 0.0:
        return math.inf

    # Denominador de la ecuación de Paschen.
    denom = (
        math.log(townsend_a * pd)
        -
        math.log(
            math.log(
                1.0 + 1.0 / gamma
            )
        )
    )

    # Evita divisiones por cero o
    # regiones físicamente inválidas.
    if denom <= 0.0:
        return math.inf

    # Ley de Paschen.
    return (
        townsend_b
        * pd
        / denom
    )
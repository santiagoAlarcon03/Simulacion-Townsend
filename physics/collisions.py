"""
Módulo de colisiones para la simulación de Townsend.

Este módulo implementa tres procesos físicos:

1. Muestreo probabilístico de colisiones usando una frecuencia
   de colisión y un intervalo temporal.

2. Dispersión de velocidades después de una colisión,
   simulando la pérdida de energía y el cambio aleatorio
   de dirección.

3. Detección geométrica de colisiones entre electrones
   y partículas neutras utilizando una distancia umbral.
"""

import numpy as np


def sample_collisions(count, frequency, dt, rng):
    """
    Determina qué partículas sufren una colisión durante
    el paso temporal actual.

    Se utiliza un modelo probabilístico basado en un
    proceso de Poisson.

    Parámetros
    ----------
    count : int
        Número de partículas.

    frequency : float
        Frecuencia de colisión (1/s).

    dt : float
        Paso temporal de la simulación.

    rng : numpy.random.Generator
        Generador de números aleatorios.

    Retorna
    -------
    ndarray(bool)
        Vector booleano donde:

        True  -> la partícula colisionó.
        False -> no colisionó.
    """

    if count <= 0:
        return np.zeros((0,), dtype=bool)

    # Probabilidad de sufrir al menos una colisión
    # durante el intervalo dt.
    probability = 1.0 - np.exp(-frequency * dt)

    # Se genera un número aleatorio para cada partícula.
    # Si el número es menor que la probabilidad,
    # se considera que ocurrió una colisión.
    return rng.random(count) < probability


def scatter_velocities(velocities, mask, rng):
    """
    Modifica las velocidades de las partículas que
    han colisionado.

    Después de la colisión:

    - La dirección se vuelve aleatoria.
    - La velocidad disminuye al 50%.
    - Se conserva la magnitud original antes
      de aplicar la pérdida de energía.

    Parámetros
    ----------
    velocities : ndarray
        Matriz Nx3 con velocidades.

    mask : ndarray(bool)
        Máscara indicando qué partículas colisionaron.

    rng : numpy.random.Generator
        Generador de números aleatorios.
    """

    if mask.size == 0 or not mask.any():
        return

    # Magnitud de la velocidad de cada partícula
    # que sufrió colisión.
    speeds = np.linalg.norm(
        velocities[mask],
        axis=1
    )

    # Generación de direcciones aleatorias 3D.
    directions = rng.normal(
        size=(mask.sum(), 3)
    )

    # Normalización de los vectores.
    directions /= (
        np.linalg.norm(
            directions,
            axis=1,
            keepdims=True
        )
        + 1.0e-12
    )

    # Nueva velocidad:
    # dirección aleatoria
    # velocidad reducida al 50%
    velocities[mask] = (
        directions
        * speeds[:, None]
        * 0.5
    )


def detect_electron_neutral_collisions(
    electron_positions,
    neutral_positions,
    radius
):
    """
    Detecta colisiones entre electrones y partículas neutras.

    Una colisión ocurre cuando la distancia entre un electrón
    y una partícula neutra es menor o igual al radio de colisión.

    Parámetros
    ----------
    electron_positions : ndarray
        Posiciones de electrones (Nx3).

    neutral_positions : ndarray
        Posiciones de partículas neutras (Mx3).

    radius : float
        Radio efectivo de colisión.

    Retorna
    -------
    tuple

        electron_mask : ndarray(bool)
            Electrones que colisionaron.

        neutral_mask : ndarray(bool)
            Neutros involucrados en colisiones.

        collision_points : ndarray
            Coordenadas donde se detectaron colisiones.
    """

    # Si no existen partículas de alguno de los tipos,
    # no pueden existir colisiones.
    if (
        electron_positions.size == 0
        or neutral_positions.size == 0
    ):
        return (
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=bool),
            np.zeros((0, 3), dtype=float),
        )

    electron_positions = np.asarray(
        electron_positions,
        dtype=float
    )

    neutral_positions = np.asarray(
        neutral_positions,
        dtype=float
    )

    # Distancia máxima permitida para considerar
    # una colisión.
    threshold_sq = float(radius) ** 2

    electron_mask = np.zeros(
        (electron_positions.shape[0],),
        dtype=bool
    )

    neutral_mask = np.zeros(
        (neutral_positions.shape[0],),
        dtype=bool
    )

    collision_points = []

    # Se analiza cada partícula neutra.
    for neutral_index, neutral_pos in enumerate(
        neutral_positions
    ):

        if electron_positions.size == 0:
            break

        # Vector diferencia entre el neutro
        # y todos los electrones.
        deltas = electron_positions - neutral_pos

        # Distancias al cuadrado.
        distances_sq = np.sum(
            deltas * deltas,
            axis=1
        )

        # Electrón más cercano.
        electron_index = int(
            np.argmin(distances_sq)
        )

        # Verificación de colisión.
        if (
            distances_sq[electron_index]
            <= threshold_sq
        ):
            electron_mask[electron_index] = True
            neutral_mask[neutral_index] = True

            collision_points.append(
                neutral_pos
            )

    if collision_points:
        return (
            electron_mask,
            neutral_mask,
            np.asarray(
                collision_points,
                dtype=float
            ),
        )

    return (
        electron_mask,
        neutral_mask,
        np.zeros((0, 3), dtype=float),
    )
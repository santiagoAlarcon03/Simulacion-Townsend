"""
Módulo de partículas.

Contiene funciones para:

- Inicializar electrones.
- Inicializar partículas neutras.
- Generar velocidades térmicas.
- Calcular aceleraciones debidas al campo eléctrico.
- Calcular energía cinética en eV.
- Estimar la corriente eléctrica producida.

Estas funciones constituyen la base física del movimiento
de partículas en la simulación de Townsend.
"""

import math
import numpy as np

from physics import constants


def initialize_electrons(count, config, rng):
    """
    Genera electrones iniciales cerca del cátodo.

    Los electrones aparecen distribuidos aleatoriamente
    en X e Y, y muy cerca del electrodo emisor (z≈0).

    Parámetros
    ----------
    count : int
        Número de electrones.

    config : object
        Configuración de la simulación.

    rng : numpy.random.Generator
        Generador aleatorio.

    Retorna
    -------
    tuple
        (positions, velocities)
    """

    positions = np.zeros((count, 3), dtype=float)

    positions[:, 0] = rng.uniform(
        -config.xy_extent,
        config.xy_extent,
        size=count
    )

    positions[:, 1] = rng.uniform(
        -config.xy_extent,
        config.xy_extent,
        size=count
    )

    positions[:, 2] = rng.uniform(
        0.0,
        config.gap_distance * 0.05,
        size=count
    )

    velocities = random_thermal_velocities(
        count,
        rng,
        config.gas_temperature
    )

    return positions, velocities


def initialize_neutral_particles(count, config, rng):
    """
    Genera partículas neutras distribuidas
    por todo el volumen de gas.

    Retorna
    -------
    tuple
        (positions, velocities)
    """

    positions = np.zeros((count, 3), dtype=float)

    positions[:, 0] = rng.uniform(
        -config.xy_extent,
        config.xy_extent,
        size=count
    )

    positions[:, 1] = rng.uniform(
        -config.xy_extent,
        config.xy_extent,
        size=count
    )

    positions[:, 2] = rng.uniform(
        0.0,
        config.gap_distance,
        size=count
    )

    velocities = random_thermal_velocities(
        count,
        rng,
        config.gas_temperature * 0.5
    )

    return positions, velocities


def random_thermal_velocities(
    count,
    rng,
    temperature=300.0,
    mass=None  # 💡 Agregamos masa parametrizable
):
    """
    Genera velocidades térmicas usando la distribución
    de Maxwell-Boltzmann adecuada para la masa de la partícula.
    """
    if mass is None:
        mass = constants.ELECTRON_MASS

    thermal_speed = math.sqrt(
        constants.BOLTZMANN
        * temperature
        / mass
    )

    return rng.normal(
        0.0,
        *thermal_speed,  # Nota: si usas la escala normal, pasa thermal_speed directamente
        size=(count, 3)
    )


def random_thermal_velocities(
    count,
    rng,
    temperature=300.0,
    mass=None
):
    """
    Genera velocidades térmicas usando la distribución
    de Maxwell-Boltzmann adecuada para la masa de la partícula.
    """
    if mass is None:
        mass = constants.ELECTRON_MASS

    thermal_speed = math.sqrt(
        constants.BOLTZMANN
        * temperature
        / mass
    )

    # 🎯 CORRECCIÓN: Quitamos el asterisco (*) antes de thermal_speed
    return rng.normal(
        0.0,
        thermal_speed,
        size=(count, 3)
    )


def acceleration_from_field(
    field_vector,
    charge,
    mass
):
    """
    Calcula la aceleración producida
    por un campo eléctrico.

    Retorna
    -------
    ndarray
        Vector aceleración.
    """

    return (
        charge / mass
    ) * np.asarray(
        field_vector,
        dtype=float
    )


def kinetic_energy_ev(velocities, mass=None):  # 💡 Agregamos masa parametrizable
    """
    Calcula energía cinética expresada en electronvoltios (eV) 
    soportando la masa real de la partícula evaluada.
    """
    if velocities.size == 0:
        return np.zeros((0,), dtype=float)

    if mass is None:
        mass = constants.ELECTRON_MASS

    speed_sq = np.sum(velocities * velocities, axis=1)

    # Multiplica por la masa correcta (Electrón o Ion N2+)
    energy_j = 0.5 * mass * speed_sq

    return energy_j / abs(constants.ELECTRON_CHARGE)


def estimate_current(
    velocities,
    charge,
    area
):
    """
    Estima la corriente eléctrica.

    Retorna
    -------
    float
        Corriente estimada.
    """

    if velocities.size == 0:
        return 0.0

    area = max(area, 1.0e-12)

    return (
        charge
        * float(
            np.sum(
                velocities[:, 2]
            )
        )
        / area
    )

def random_directional_velocities(
    count,
    rng,
    temperature=300.0,
    mass=None
):
    """
    Genera velocidades con dirección aleatoria
    y magnitud térmica.

    Las direcciones se distribuyen uniformemente
    sobre una esfera.
    """

    if count == 0:
        return np.zeros((0, 3), dtype=float)

    if mass is None:
        mass = constants.ELECTRON_MASS

    thermal_speed = math.sqrt(
        constants.BOLTZMANN
        * temperature
        / mass
    )

    dirs = rng.normal(
        size=(count, 3)
    )

    norms = np.linalg.norm(
        dirs,
        axis=1
    )

    norms = np.where(
        norms == 0.0,
        1.0,
        norms
    )

    unit_dirs = dirs / norms[:, None]

    return unit_dirs * thermal_speed
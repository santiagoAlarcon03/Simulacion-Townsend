from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class Stage(Enum):
    INITIAL_ELECTRONS = auto()
    FIELD_ACCELERATION = auto()
    COLLISIONS = auto()
    IONIZATION = auto()
    AVALANCHE = auto()
    EXPONENTIAL_GROWTH = auto()
    SELF_SUSTAINED = auto()
    CHARGE_EVOLUTION = auto()
    VISUALIZATION = auto()


STAGE_ORDER = [
    Stage.INITIAL_ELECTRONS,
    Stage.FIELD_ACCELERATION,
    Stage.COLLISIONS,
    Stage.IONIZATION,
    Stage.AVALANCHE,
    Stage.EXPONENTIAL_GROWTH,
    Stage.SELF_SUSTAINED,
    Stage.CHARGE_EVOLUTION,
    Stage.VISUALIZATION,
]


STAGE_LABELS = {
    Stage.INITIAL_ELECTRONS: "Initial electrons",
    Stage.FIELD_ACCELERATION: "Field acceleration",
    Stage.COLLISIONS: "Collisions with gas",
    Stage.IONIZATION: "Impact ionization",
    Stage.AVALANCHE: "Electron avalanche",
    Stage.EXPONENTIAL_GROWTH: "Exponential growth",
    Stage.SELF_SUSTAINED: "Self sustained discharge",
    Stage.CHARGE_EVOLUTION: "Charge and current evolution",
    Stage.VISUALIZATION: "Visualization",
}

from dataclasses import dataclass, field
import numpy as np
from simulation.state import Stage  # O la ruta correspondiente en tu proyecto


@dataclass
class SimulationState:
    """Contenedor de estado dinámico para la simulación física de descargas eléctricas.
    
    Esta clase almacena las propiedades macroscópicas (tiempo, etapa actual) y las 
    propiedades microscópicas (posiciones y velocidades vectoriales en 3D) de todas 
    las especies de partículas presentes en el sistema: electrones, neutros e iones.
    """

    # --- ESPECIE: ELECTRONES ---
    positions: np.ndarray
    """np.ndarray: Matriz de forma (N, 3) con las coordenadas cartesianas [x, y, z] 
    de los N electrones libres en el dominio."""

    velocities: np.ndarray
    """np.ndarray: Matriz de forma (N, 3) con los vectores de velocidad [vx, vy, vz] 
    de los N electrones libres (en m/s)."""

    # --- PARÁMETROS GLOBALES DEL SISTEMA ---
    time: float
    """float: Tiempo físico acumulado desde el inicio de la simulación (en segundos)."""

    stage: Stage
    """Stage: Instancia de enumeración que representa la fase física actual de la descarga 
    (ej. Colisiones, Avalancha, Crecimiento Exponencial)."""

    # --- CONTADORES DE EVENTOS STOCHÁSTICOS (POR PASO DE TIEMPO) ---
    collision_events: int = 0
    """int: Cantidad acumulada de colisiones elásticas ocurridas en el último paso temporal."""

    ionization_events: int = 0
    """int: Cantidad acumulada de eventos de ionización por impacto ocurridos en el último paso."""

    # --- ESPECIE: GAS NEUTRO ---
    neutral_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    """np.ndarray: Matriz de forma (M, 3) con las posiciones [x, y, z] de las M partículas 
    del gas de fondo neutro."""

    neutral_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    """np.ndarray: Matriz de forma (M, 3) con los vectores de velocidad [vx, vy, vz] 
    de las M partículas del gas neutro."""

    # --- ESPECIE: IONES (NUEVO) ---
    ion_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    """np.ndarray: Matriz de forma (K, 3) con las posiciones [x, y, z] de los K iones positivos 
    generados tras los procesos de ionización."""

    ion_velocities: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float)
    )
    """np.ndarray: Matriz de forma (K, 3) con los vectores de velocidad [vx, vy, vz] 
    de los K iones positivos. Se desplazarán en sentido opuesto a los electrones debido a su carga."""
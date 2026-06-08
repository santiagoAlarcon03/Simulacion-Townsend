"""
Módulo de registro de datos.

La clase DataLogger almacena información generada durante
la simulación para su posterior análisis.

Actualmente registra:

- Tiempo de simulación.
- Número de electrones.
- Número de iones positivos.
- Corriente eléctrica.

Estos datos pueden utilizarse para generar gráficas
o exportar resultados.
"""


class DataLogger:
    """
    Registrador de datos de la simulación.

    Mantiene listas con la evolución temporal
    de las principales variables físicas.
    """

    def __init__(self) -> None:
        """
        Inicializa el logger.

        Al crearse, se vacían todos los registros.
        """
        self.reset()

    def reset(self) -> None:
        """
        Elimina todos los datos almacenados.

        Se utiliza normalmente cuando se reinicia
        la simulación.
        """

        # Tiempo transcurrido (s)
        self.times = []

        # Número de electrones
        self.counts = []

        # Número de iones positivos
        self.ion_counts = []

        # Corriente eléctrica (A)
        self.currents = []

    def log(
        self,
        time_s: float,
        count: int,
        current_a: float,
        ion_count: int = 0,
    ) -> None:
        """
        Registra un nuevo punto de datos.

        Parámetros
        ----------
        time_s : float
            Tiempo de simulación en segundos.

        count : int
            Número de electrones existentes.

        current_a : float
            Corriente instantánea en amperios.

        ion_count : int
            Número de iones positivos presentes.
        """

        self.times.append(float(time_s))
        self.counts.append(int(count))
        self.ion_counts.append(int(ion_count))
        self.currents.append(float(current_a))
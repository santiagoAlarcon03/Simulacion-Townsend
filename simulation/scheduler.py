"""
Módulo de planificación temporal.

La clase SimulationScheduler controla la ejecución periódica
de la simulación.

Su función principal es llamar repetidamente al motor de
simulación utilizando un paso temporal fijo (dt).

Actúa como un reloj que determina cuándo debe actualizarse
el estado físico del sistema.
"""


class SimulationScheduler:
    """
    Programador temporal de la simulación.

    Mantiene:

    - Paso temporal (dt).
    - Función de actualización.
    - Estado de ejecución.
    """

    def __init__(self, dt, callback) -> None:
        """
        Inicializa el scheduler.

        Parámetros
        ----------
        dt : float
            Paso temporal de simulación.

        callback : callable
            Función que será ejecutada en cada tick.
            Generalmente corresponde a:

                engine.step(dt)
        """

        self.dt = dt

        # Función que ejecutará la actualización.
        self.callback = callback

        # Estado inicial.
        self.running = False

    def start(self) -> None:
        """
        Inicia la simulación.

        Después de llamar a este método,
        cada tick ejecutará el callback.
        """

        self.running = True

    def stop(self) -> None:
        """
        Detiene la simulación.

        Los ticks seguirán ocurriendo,
        pero no se ejecutará el callback.
        """

        self.running = False

    def tick(self) -> None:
        """
        Ejecuta un ciclo temporal.

        Si la simulación está activa:

            callback(dt)

        En caso contrario no realiza ninguna acción.
        """

        if self.running:
            self.callback(self.dt)
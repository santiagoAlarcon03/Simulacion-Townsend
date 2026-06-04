import unittest

from config import SimulationConfig
from simulation.engine import SimulationEngine
from simulation.state import Stage


class EngineTests(unittest.TestCase):
    def test_step(self):
        config = SimulationConfig()
        engine = SimulationEngine(config)
        engine.reset(Stage.INITIAL_ELECTRONS, 10)
        state, metrics = engine.step(config.dt)
        self.assertIsNotNone(state)
        self.assertIn("count", metrics)


if __name__ == "__main__":
    unittest.main()

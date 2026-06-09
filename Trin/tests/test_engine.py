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

    def test_add_electron_increases_count(self):
        config = SimulationConfig()
        engine = SimulationEngine(config)
        engine.reset(Stage.INITIAL_ELECTRONS, 2)
        before = engine.state.positions.shape[0]
        engine.add_electron()
        after = engine.state.positions.shape[0]
        self.assertEqual(after, before + 1)
        self.assertGreaterEqual(engine.state.positions.shape[0], 3)
        self.assertGreaterEqual(engine.state.positions[-1, 2], config.gap_distance * 0.7)

    def test_particles_remain_visible_with_reflection(self):
        config = SimulationConfig()
        engine = SimulationEngine(config)
        engine.reset(Stage.INITIAL_ELECTRONS, 2)
        # Only test lateral (X/Y) reflection; Z boundaries are now absorbing electrodes
        engine.state.positions[:] = 0.0
        engine.state.positions[0, 0] = config.xy_extent * 1.2  # Over X boundary
        engine.state.velocities[:] = 0.0
        engine.state.neutral_positions = engine.state.neutral_positions[:0]
        state, _ = engine.step(config.dt)
        self.assertIsNotNone(state)
        # Both electrons remain because they did not hit the anode
        self.assertEqual(state.positions.shape[0], 2)

    def test_add_neutral_particle(self):
        config = SimulationConfig()
        engine = SimulationEngine(config)
        engine.reset(Stage.INITIAL_ELECTRONS, 2)
        before = engine.state.neutral_positions.shape[0]
        engine.add_neutral()
        after = engine.state.neutral_positions.shape[0]
        self.assertEqual(after, before + 1)
        self.assertEqual(engine.state.positions.shape[0], 2)

    def test_electron_neutral_collision_spawns_electron(self):
        config = SimulationConfig()
        engine = SimulationEngine(config)
        engine.reset(Stage.INITIAL_ELECTRONS, 1)
        engine.state.positions[0] = 0.0
        engine.state.velocities[0] = 0.0
        engine.state.neutral_positions = engine.state.positions.copy()
        engine.state.neutral_velocities = engine.state.velocities.copy()
        before = engine.state.positions.shape[0]
        state, _ = engine.step(config.dt)
        self.assertIsNotNone(state)
        self.assertGreater(state.positions.shape[0], before)
        # Verify that a positive ion is also created
        self.assertEqual(state.ion_positions.shape[0], 1)


if __name__ == "__main__":
    unittest.main()

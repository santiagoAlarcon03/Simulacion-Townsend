# Methodology

The simulation uses a particle approach with explicit time stepping.
At each step:
1. Update velocities by the electric field.
2. Move particles.
3. Sample collisions and apply scattering.
4. Sample ionization and create new electrons.
5. Update stage and metrics.

This is intended as a clear, teachable model, not an industrial plasma solver.

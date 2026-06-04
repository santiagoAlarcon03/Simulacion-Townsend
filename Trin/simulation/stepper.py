def euler_step(positions, velocities, acceleration, dt):
    velocities = velocities + acceleration * dt
    positions = positions + velocities * dt
    return positions, velocities

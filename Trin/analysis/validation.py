def validate_dt(dt):
    return 0.0 < float(dt) < 1.0e-6


def validate_particle_count(count, max_count):
    return 0 < int(count) <= int(max_count)

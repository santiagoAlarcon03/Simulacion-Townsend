import math

from physics import avalanche


def is_self_sustained(config):
    alpha = avalanche.townsend_alpha(
        config.townsend_A, config.townsend_B, config.electric_field, config.gas_pressure
    )
    if alpha <= 0.0:
        return False
    multiplier = math.exp(alpha * config.gap_distance) - 1.0
    return config.secondary_emission_gamma * multiplier >= 1.0

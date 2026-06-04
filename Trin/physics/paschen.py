import math


def paschen_voltage(pressure, distance, townsend_a, townsend_b, gamma=0.02):
    pd = pressure * distance
    if pd <= 0.0:
        return math.inf
    denom = math.log(townsend_a * pd) - math.log(math.log(1.0 + 1.0 / gamma))
    if denom <= 0.0:
        return math.inf
    return townsend_b * pd / denom

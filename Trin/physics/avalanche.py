import math


def townsend_alpha(townsend_a, townsend_b, electric_field, pressure):
    if electric_field <= 0.0 or pressure <= 0.0:
        return 0.0
    return townsend_a * pressure * math.exp(-townsend_b * pressure / electric_field)


def multiplication_factor(alpha, distance):
    return math.exp(alpha * distance)

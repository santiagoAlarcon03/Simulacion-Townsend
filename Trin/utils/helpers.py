def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def safe_div(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return numerator / denominator

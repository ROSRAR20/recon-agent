"""Mathematical and integer rounding utilities for recon-agent synthetic generation.

Ensures strict minor currency unit (integer paise) arithmetic without floats.
"""


def mul_div_round(x: int, num: int, den: int) -> int:
    """Integer half-up multiplication and division: round(x * num / den).

    Computes half-up rounding using pure integer arithmetic:
    (2 * x * num + den) // (2 * den) for non-negative values.

    Args:
        x: Base integer value (e.g., amount in paise).
        num: Multiplier (e.g., basis points).
        den: Divisor (e.g., 10,000 for basis points).

    Returns:
        Integer rounded half-up.

    Raises:
        ZeroDivisionError: If den is 0.
        ValueError: If x, num, or den is negative.
    """
    if den == 0:
        raise ZeroDivisionError("den cannot be zero")
    if x < 0 or num < 0 or den < 0:
        raise ValueError("mul_div_round expects non-negative integers")

    return (2 * x * num + den) // (2 * den)

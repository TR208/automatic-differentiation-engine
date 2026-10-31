"""Member 4: analytical validation and forward finite-difference experiments.

Run: python -X utf8 -B testing.py
"""

import math
import numpy as np
from dual import Dual


def polynomial(x):
    return x**3 - 2*x + 1


def polynomial_derivative(x):
    return 3*x**2 - 2


def composed_function(x):
    # Use Dual methods for AD and math functions for ordinary numerical evaluation.
    if isinstance(x, Dual):
        return (1 + x**2) * x.sin() * x.exp()
    return (1 + x**2) * math.sin(x) * math.exp(x)


def composed_derivative(x):
    # Product rule applied to (1 + x^2), sin(x), and exp(x).
    return math.exp(x) * (
        2*x*math.sin(x) + (1 + x**2)*(math.cos(x) + math.sin(x))
    )


def logarithmic_function(x):
    # x^2 + 1 is positive for every real x.
    if isinstance(x, Dual):
        return (x**2 + 1).log()
    return math.log(x**2 + 1)


def logarithmic_derivative(x):
    return 2*x / (x**2 + 1)


def evaluate_ad(function, point):
    """Reuse the same AD procedure for any of the example functions."""
    x = Dual(point, 1)
    return function(x)


def forward_difference(function, point, h):
    return (function(point + h) - function(point)) / h


def check(name, calculation, expected):
    """Compare both the function value and its derivative with known answers."""
    try:
        result = calculation()
        actual = (result.value, result.derivative)
        if np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
            print(f"PASS: {name}")
            return True
        print(f"FAIL: {name}: got {actual}, expected {expected}")
    except Exception as error:
        print(f"FAIL: {name}: {type(error).__name__}: {error}")
    return False


def check_error(name, calculation, expected_error):
    """An invalid operation should raise the expected exception."""
    try:
        calculation()
    except expected_error:
        print(f"PASS: {name}")
        return True
    except Exception as error:
        print(f"FAIL: {name}: got {type(error).__name__}, expected {expected_error.__name__}")
        return False
    print(f"FAIL: {name}: expected {expected_error.__name__}, but no exception was raised")
    return False


if __name__ == "__main__":
    print("\nMember 4 verification")
    # Each expected pair is (value, derivative). Seed 1 differentiates with respect to x.
    arithmetic = [
        ("addition", lambda: Dual(2, 1) + Dual(3, 0), (5, 1)),
        ("subtraction", lambda: Dual(2, 1) - Dual(3, 0), (-1, 1)),
        ("multiplication", lambda: Dual(2, 1) * Dual(3, 2), (6, 7)),
        ("division", lambda: Dual(6, 1) / Dual(2, 0), (3, 0.5)),
        ("reverse subtraction", lambda: 5 - Dual(2, 1), (3, -1)),
        ("reverse division", lambda: 2 / Dual(4, 1), (0.5, -0.125)),
        ("zero derivative seed", lambda: 2 / Dual(4, 0), (0.5, 0)),
        ("scaled derivative seed", lambda: Dual(2, 3)**2, (4, 12)),
        ("reverse power", lambda: 3**Dual(2, 1), (9, 9 * math.log(3))),
    ]
    results = [check(name, calculation, expected)
               for name, calculation, expected in arithmetic]

    # Task 1: analytical derivatives are references only; evaluate_ad does not use them.
    functions = [
        ("x^3 - 2x + 1", polynomial, polynomial_derivative),
        ("(1 + x^2) sin(x) exp(x)", composed_function, composed_derivative),
        ("log(x^2 + 1)", logarithmic_function, logarithmic_derivative),
    ]
    analytical_results = []
    print("\nTask 1: comparison with analytical derivatives")
    print("Numerical tolerance: rtol=1e-10, atol=1e-12")
    for name, function, exact_derivative in functions:
        print(f"\nf(x) = {name}")
        print(f"{'x':>6} {'AD derivative':>18} {'Analytical':>18} {'Absolute error':>16} {'Result':>8}")
        for point in (-0.5, 0.0, 2.0):
            result = evaluate_ad(function, point)
            exact = exact_derivative(point)
            error = abs(result.derivative - exact)
            passed = np.allclose(
                (result.value, result.derivative), (function(point), exact),
                rtol=1e-10, atol=1e-12
            )
            analytical_results.append(passed)
            status = "PASS" if passed else "FAIL"
            print(f"{point:6.1f} {result.derivative:18.10f} {exact:18.10f} {error:16.3e} {status:>8}")
    results.extend(analytical_results)
    print(f"\nAnalytical comparisons: {sum(analytical_results)}/{len(analytical_results)} passed.")

    # Task 2: the same polynomial works with floats for FD and Dual objects for AD.
    point = 2.0
    exact = polynomial_derivative(point)
    ad_derivative = evaluate_ad(polynomial, point).derivative
    print("\nTask 2: forward finite differences for x^3 - 2x + 1 at x=2")
    print(f"Analytical derivative: {exact:.10f}")
    print(f"AD derivative:         {ad_derivative:.10f}")
    print(f"AD absolute error:     {abs(ad_derivative - exact):.3e}")
    print(f"\n{'h':>12} {'FD derivative':>18} {'FD absolute error':>20}")
    best_error = math.inf
    best_h = None
    for exponent in range(1, 17):
        h = 10.0**(-exponent)
        approximation = forward_difference(polynomial, point, h)
        error = abs(approximation - exact)
        print(f"{h:12.1e} {approximation:18.10f} {error:20.3e}")
        if error < best_error:
            best_error = error
            best_h = h
    print(f"\nSmallest FD error among tested steps: {best_error:.3e} at h={best_h:.1e}")
    # In exact arithmetic, the FD derivative here is 10 + 6h + h^2.
    print("Truncation: the exact-arithmetic FD error is 6h + h^2, so it decreases like h.")
    print("Cancellation: for tiny h, subtracting nearly equal function values loses precision.")
    print(f"At h=1e-16, x+h == x is {point + 1e-16 == point}; the computed FD numerator is zero.")
    print("AD uses no step size and avoids this finite-difference subtraction.")

    invalid_inputs = [
        ("division by a zero Dual", lambda: Dual(1, 1) / Dual(0, 1), ZeroDivisionError),
        ("reverse division by zero", lambda: 1 / Dual(0, 1), ZeroDivisionError),
        ("division by numeric zero", lambda: Dual(1, 1) / 0, ZeroDivisionError),
        ("log(0)", lambda: Dual(0, 1).log(), ValueError),
        ("log(-1)", lambda: Dual(-1, 1).log(), ValueError),
        ("zero exponent at zero", lambda: Dual(0, 1)**0, ValueError),
    ]
    results.extend(check_error(name, calculation, expected_error)
                   for name, calculation, expected_error in invalid_inputs)

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    raise SystemExit(0 if all(results) else 1)

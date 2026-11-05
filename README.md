# Automatic Differentiation Engine

Scientific Computing Project 6 implements a small forward-mode automatic differentiation (AD) engine in Python. A `Dual` object carries a function value and a derivative component together. Arithmetic operators and elementary functions propagate both through a calculation.

The project compares AD with analytical derivatives and forward finite differences. The [report](REPORT.md) contains the derivations, a worked chain-rule trace, measured results, and limitations.

## Project files

| File | Purpose |
|---|---|
| `dual.py` | The `Dual` class, operator rules, elementary functions, and direct-run demonstrations |
| `testing.py` | Reusable function examples, analytical checks, invalid-input checks, and a finite-difference experiment |
| `requirements.txt` | The NumPy dependency |
| `REPORT.md` | The project experiments and their interpretation |
| `.gitignore` | Excludes Python bytecode and the local virtual environment |

## Setup and execution

Use Python with NumPy installed. The documented results were verified using Python **3.12.14** and NumPy **2.3.5** on 3 November 2026. Other environments may differ in the last few floating-point digits.

From the repository directory, these PowerShell commands create a local environment and install the dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the demonstrations and verification separately:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B dual.py
.\.venv\Scripts\python.exe -X utf8 -B testing.py
```

If NumPy is already installed in your active Python environment, use `python -X utf8 -B testing.py` directly. UTF-8 supports the epsilon symbol in the demonstrations; `-B` avoids writing bytecode files.

## Calculate a value and derivative

For `f(x) = x**2 + 3*x + 1`, evaluated at `x = 2`:

```python
from dual import Dual

def my_function(x):
    return x**2 + 3*x + 1

x = Dual(2.0, 1.0)
result = my_function(x)

print(result.value)       # 11.0
print(result.derivative)  # 7.0
```

The constructor is `Dual(value, derivative)`. The seed `1` represents `dx/dx = 1`; the engine then computes the derivative of the complete expression. A seed of `0` treats that input as constant. A seed of `c` scales the propagated derivative by `c`.

Importing `Dual` does not run the demonstrations. They are inside `if __name__ == "__main__":` and run when `dual.py` is executed directly.

## Reuse the interface for different functions

`evaluate_ad(function, point)` in `testing.py` constructs `Dual(point, 1)` and calls the supplied function. It returns a `Dual` result and requires no analytical derivative formula.

```python
from testing import evaluate_ad, composed_function, logarithmic_function

first = evaluate_ad(composed_function, 2.0)
second = evaluate_ad(logarithmic_function, 2.0)

print(first.value, first.derivative)    # About 33.5942484871, 45.0949856737
print(second.value, second.derivative)  # About 1.6094379124, 0.8
```

Users define a Python function using supported operations. For a `Dual` input, use methods such as `x.sin()` and `x.exp()` instead of `math.sin(x)` and `math.exp(x)`. The examples in `testing.py` explicitly select the Dual methods or ordinary `math` functions so the same example function also accepts a normal number.

The functions named `polynomial_derivative`, `composed_derivative`, and `logarithmic_derivative` provide independent reference answers for verification. `evaluate_ad()` does not call them.

## Supported operations and domains

| Operation | Interface and behavior |
|---|---|
| Arithmetic | `+`, `-`, `*`, `/`, `**`, including implemented reverse operations such as `2/x` and `3**x` |
| Elementary functions | `x.sin()`, `x.cos()`, `x.tan()`, `x.exp()`, `x.log()` |
| Logarithm | Restricted to positive real values; nonpositive values raise `ValueError`, and complex values raise `TypeError` |
| Tangent | Rejects inputs whose cosine is numerically close to zero, using `atol=1e-15` |
| Zero to the power zero | The project explicitly rejects `0**0` with `ValueError` |
| Unsupported operand types | Operator methods raise `TypeError` |
| Division by zero | Ordinary Python-zero cases raise `ZeroDivisionError`; see the NumPy limitation below |

The verified examples use real scalar inputs. For variable exponents and constant-base exponentiation, use positive bases in the demonstrated real-valued workflow. These methods currently use a complex logarithm internally, so a mathematically real derivative can be represented as `a+0j`. Comprehensive complex-domain behavior is not established by the current tests.

## Verification and experiments

Running `testing.py` prints:

- Nine arithmetic and derivative-seed checks.
- Tables comparing three functions with analytical derivatives at `-0.5`, `0.0`, and `2.0`.
- A forward finite-difference table for `x**3 - 2*x + 1` at `x=2`, using 16 step sizes from `1e-1` through `1e-16`.
- Six checks expecting exceptions for invalid inputs.

The verified run reports **24/24 checks passed**, including **9/9 analytical comparisons**. Numerical comparisons use `rtol=1e-10` and `atol=1e-12` for both values and derivatives. The program exits with code `0` when all checks pass and `1` if a check fails or execution stops with an unhandled exception.

The finite-difference rows are an accuracy experiment, not 16 additional pass/fail tests: large errors at tiny step sizes illustrate cancellation and rounding. In the recorded run, the smallest tested FD error was about `6.077e-8` at `h=1e-8`; the AD derivative was `10.0`. See [the report](REPORT.md) for the complete tables and explanation.

## Known limitations

- A zero denominator stored as a NumPy scalar can produce infinity and warnings instead of `ZeroDivisionError`. For example, `1 / Dual(0, 1).sin()` returns an infinite value. The 24 checks do not cover this case; explicit denominator checks remain a needed engine correction.
- The scope is first derivatives of supported scalar expressions. The interface does not accept mathematical expressions as strings or automatically differentiate arbitrary external Python functions.
- Floating-point AD can still accumulate rounding error or overflow. It avoids the step-size approximation of finite differences, but is not exact real arithmetic.
- The old expected-answer comment for `sin(exp(2))` in `dual.py` is inaccurate. Its computed value and derivative are about `0.8938549549` and `3.3129294231`; the report uses the computed results.

## Team work and project brief

The allocation in the course brief assigns the Dual core to Member 1, arithmetic to Member 2, elementary functions and domains to Member 3, and verification and demonstrations to Member 4. This repository's implementation and experiments follow that division.

The report maps the four required experiments to the code. GitHub issues, feature branches, PR discussions, and peer reviews supply the collaboration evidence. The team should retain each member's review of another member's work as required by the brief.

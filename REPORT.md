# Automatic Differentiation Engine Project Report

Scientific Computing Project 6

This report documents the shared forward-mode AD engine and Member 4's verification and demonstrations. It follows the experiments in the supplied course brief, `Project_6_Automatic_Differentiation_Engine.docx`.

The reported experiments use **Python 3.12.14** and **NumPy 2.3.5**.

## 1 Objective and implementation

The objective is to evaluate a scalar function and its first derivative by executing the function with a value that also carries derivative information. `dual.py` defines `Dual(value, derivative)`, overloads arithmetic operations, and implements `sin`, `cos`, `tan`, `exp`, and `log` methods. `testing.py` exercises the engine and compares its results with analytical derivatives and forward finite differences.

A Dual quantity can be written as $a + a'\varepsilon$, where $\varepsilon^2=0$. The epsilon here is a formal dual-number symbol, not floating-point machine epsilon. For instance,

$$
(a+a'\varepsilon)(b+b'\varepsilon)=ab+(a'b+ab')\varepsilon.
$$

Thus the multiplication overload calculates both the product value and its derivative using the product rule. Other operations apply their corresponding local rules. The class stores the two components directly; it does not store symbolic expressions.

| Operation | Value | Propagated derivative |
|---|---|---|
| $a+b$ | $a+b$ | $a'+b'$ |
| $a-b$ | $a-b$ | $a'-b'$ |
| $ab$ | $ab$ | $a'b+ab'$ |
| $a/b$ | $a/b$ | $(a'b-ab')/b^2$, for a nonzero denominator |
| $a^n$ with constant $n$ | $a^n$ | $na^{n-1}a'$ where the expression is defined |
| $\sin(a)$ | $\sin(a)$ | $\cos(a)a'$ |
| $\cos(a)$ | $\cos(a)$ | $-\sin(a)a'$ |
| $\exp(a)$ | $\exp(a)$ | $\exp(a)a'$ |
| $\log(a)$ | $\log(a)$ | $a'/a$, for positive real $a$ |

Seeding `Dual(x0, 1)` starts with $dx/dx=1$ at $x=x_0$. After evaluation, `.value` contains $f(x_0)$ and `.derivative` contains $f'(x_0)$. Constants have zero derivative. A nonunit input seed multiplies the propagated derivative by that seed; it changes the starting rate of change rather than the function value.

## 2 Analytical validation

Three functions were selected to exercise arithmetic, products, and elementary-function composition:

| Function | Analytical derivative |
|---|---|
| $f_1(x)=x^3-2x+1$ | $f_1'(x)=3x^2-2$ |
| $f_2(x)=(1+x^2)\sin(x)\exp(x)$ | $f_2'(x)=\exp(x)[2x\sin(x)+(1+x^2)(\cos(x)+\sin(x))]$ |
| $f_3(x)=\log(1+x^2)$ | $f_3'(x)=2x/(1+x^2)$ |

The second derivative formula follows by differentiating each of the three factors in turn and adding the products. For the third function, the outer logarithm contributes $1/(1+x^2)$ and the inner expression contributes $2x$. Its logarithm argument is positive for every real input.

Each function was evaluated at $x=-0.5$, $0$, and $2$, with seed `1`. Values and derivatives were checked using `numpy.allclose`, with `rtol=1e-10` and `atol=1e-12`. Componentwise, the permitted difference is `atol + rtol * abs(reference)`. The absolute tolerance matters when the reference is zero.

| Function | Point | AD derivative | Analytical derivative | Absolute derivative error |
|---|---:|---:|---:|---:|
| $f_1$ | -0.5 | -1.2500000000 | -1.2500000000 | 0 |
| $f_1$ | 0.0 | -2.0000000000 | -2.0000000000 | 0 |
| $f_1$ | 2.0 | 10.0000000000 | 10.0000000000 | 0 |
| $f_2$ | -0.5 | 0.5926543407 | 0.5926543407 | 1.110e-16 |
| $f_2$ | 0.0 | 1.0000000000 | 1.0000000000 | 0 |
| $f_2$ | 2.0 | 45.0949856737 | 45.0949856737 | 0 |
| $f_3$ | -0.5 | -0.8000000000 | -0.8000000000 | 0 |
| $f_3$ | 0.0 | 0.0000000000 | 0.0000000000 | 0 |
| $f_3$ | 2.0 | 0.8000000000 | 0.8000000000 | 0 |

All **9/9 analytical comparisons passed**. Errors are computed before the displayed derivatives are rounded. A zero error here means equality of the computed floating-point results in this run; it is not a claim of exact real-number differentiation for every input.

## 3 More involved composition and automatic chain rule

The function $f_2(x)=(1+x^2)\sin(x)\exp(x)$ combines a power, products, and two elementary functions. It is implemented as:

```python
def composed_function(x):
    if isinstance(x, Dual):
        return (1 + x**2) * x.sin() * x.exp()
    return (1 + x**2) * math.sin(x) * math.exp(x)
```

The `Dual` path executes the following steps. Each derivative is with respect to the original input $x$.

| Step | Value expression | Derivative propagation rule |
|---|---|---|
| Input | $x$ | $x'=1$ |
| Power | $u=x^2$ | $u'=2xx'$ |
| Addition | $v=1+u$ | $v'=u'$ because the constant has derivative zero |
| Sine | $s=\sin(x)$ | $s'=\cos(x)x'$ |
| Exponential | $e=\exp(x)$ | $e'=\exp(x)x'$ |
| First product | $p=vs$ | $p'=v's+vs'$ |
| Final product | $f=pe$ | $f'=p'e+pe'$ |

At $x=2$, a numerical trace gives:

| Quantity | Value | Derivative component |
|---|---:|---:|
| $x$ | 2.0000000000 | 1.0000000000 |
| $u$ | 4.0000000000 | 4.0000000000 |
| $v$ | 5.0000000000 | 4.0000000000 |
| $s$ | 0.9092974268 | -0.4161468365 |
| $e$ | 7.3890560989 | 7.3890560989 |
| $p$ | 4.5464871341 | 1.5564555246 |
| $f$ | 33.5942484871 | 45.0949856737 |

The user writes the value expression. The overloaded operations supply these local derivative rules and pass the resulting derivative components to later operations. This is how the complete derivative is assembled automatically. No function-specific derivative expression is used on this AD path.

The logarithmic example shows a nested chain rule directly: for $u=x^2+1$ and $f=\log(u)$, the power and addition produce $u'=2x$, and `.log()` produces $f'=u'/u$.

## 4 Reusable function interface

The same helper evaluates each user-defined example:

```python
def evaluate_ad(function, point):
    x = Dual(point, 1)
    return function(x)
```

For example:

```python
result = evaluate_ad(composed_function, 2.0)
print(result.value)       # About 33.5942484871
print(result.derivative)  # About 45.0949856737

result = evaluate_ad(logarithmic_function, 2.0)
print(result.value)       # About 1.6094379124
print(result.derivative)  # 0.8
```

The helper accepts a function and an evaluation point, not a derivative formula. To use another function, the user supplies another Python function built from the supported operators and methods. The analytical derivative functions in `testing.py` are separate reference calculations used to test the engine.

The examples explicitly handle ordinary numbers with `math` functions and Dual inputs with the engine's methods. This lets the same function be used for AD, ordinary value evaluation, and finite differences where needed. This interface supports scalar functions of one input; it is not a parser for arbitrary mathematical strings.

## 5 Forward finite differences

For a smooth function, the forward approximation is

$$
D_h^+f(x) = \frac{f(x+h)-f(x)}{h}, \qquad h>0.
$$

The implementation is:

```python
def forward_difference(function, point, h):
    return (function(point + h) - function(point)) / h
```

The experiment uses $f_1(x)=x^3-2x+1$ at $x=2$. The analytical derivative is $10$, and AD returns $10.0$ with zero measured error in this run. AD is evaluated once because it has no step-size parameter. Sixteen FD steps from $10^{-1}$ to $10^{-16}$ are compared with the same analytical reference.

| Step size $h$ | FD derivative | Absolute FD error |
|---:|---:|---:|
| 1e-1 | 10.6100000000 | 6.100e-1 |
| 1e-2 | 10.0601000000 | 6.010e-2 |
| 1e-3 | 10.0060010000 | 6.001e-3 |
| 1e-4 | 10.0006000100 | 6.000e-4 |
| 1e-5 | 10.0000600002 | 6.000e-5 |
| 1e-6 | 10.0000060019 | 6.002e-6 |
| 1e-7 | 10.0000005876 | 5.876e-7 |
| 1e-8 | 9.9999999392 | 6.077e-8 |
| 1e-9 | 10.0000008274 | 8.274e-7 |
| 1e-10 | 10.0000008274 | 8.274e-7 |
| 1e-11 | 10.0000008274 | 8.274e-7 |
| 1e-12 | 10.0008890058 | 8.890e-4 |
| 1e-13 | 9.9920072216 | 7.993e-3 |
| 1e-14 | 10.2140518266 | 2.141e-1 |
| 1e-15 | 8.8817841970 | 1.118e0 |
| 1e-16 | 0.0000000000 | 1.000e1 |

### Truncation error

Taylor expansion gives $D_h^+f(x)=f'(x)+h f''(x)/2+O(h^2)$. For this particular polynomial at $x=2$, exact arithmetic gives $D_h^+f_1(2)=10+6h+h^2$. Thus the truncation error is $6h+h^2$, which initially decreases approximately tenfold when $h$ decreases tenfold.

### Cancellation and rounding

When $h$ is tiny, the two function values being subtracted are nearly equal. Floating-point rounding in these values can dominate their small difference, and division by $h$ amplifies that error. A useful qualitative model is an $O(h)$ truncation term plus a roundoff term of scale $O(u/h)$, with constants depending on the function and point; $u$ denotes machine precision. For binary64, the spacing at one is approximately $2.22\times10^{-16}$.

The smallest error among the tested steps was approximately **6.077e-8 at h=1e-8**. Below that step, error generally grows or plateaus rather than continuing to decrease. This is an observation for this function, point, and environment, not a universal optimum.

At $h=10^{-16}$, the computed number `2.0 + h` equals `2.0`, so the two function evaluations are identical. Their difference is zero, producing an FD derivative of zero and an absolute error of ten. AD avoids this finite-difference subtraction and its step-size tradeoff, although its operations remain subject to floating-point rounding and overflow.

## 6 Verification scope and invalid inputs

Run `python -X utf8 -B demo.py` for a standalone demonstration of the composed function at a chosen point. It prints the function value, AD derivative, analytical reference, and finite-difference approximation, checks the value and derivative, and then runs the full `testing.py` verification suite.

The verification suite reports **24/24 checks passed**: nine arithmetic/seed cases, nine analytical function cases, and six expected-exception cases. The FD table is an experiment and is not counted as 16 additional tests.

| Expected-exception case | Expected outcome |
|---|---|
| `Dual(1, 1) / Dual(0, 1)` | `ZeroDivisionError` |
| `1 / Dual(0, 1)` | `ZeroDivisionError` |
| `Dual(1, 1) / 0` | `ZeroDivisionError` |
| `Dual(0, 1).log()` | `ValueError` |
| `Dual(-1, 1).log()` | `ValueError` |
| `Dual(0, 1)**0` | `ValueError`, following the project's chosen policy |

`check()` compares a returned value/derivative pair with a reference pair and records failure if the result disagrees or an exception occurs. `check_error()` records success only if the requested exception is raised. The script returns a nonzero exit code if a recorded check fails; a failed check is evidence to investigate, not a reason to change a correct reference answer.

Unsupported operands are rejected by the operator methods with `TypeError`; for example, `Dual(2, 1) + "hello"` raises it. This behavior was checked separately while preparing the documentation, but is not included in the 24 automated checks. The logarithm rejects complex values and requires a positive real argument. Tangent checks whether cosine is numerically near zero. Power cases outside the verified real-valued examples need a clearly defined domain policy; the implementation does not establish comprehensive complex-domain support.

One known division issue remains: `1 / Dual(0, 1).sin()` uses a NumPy zero denominator and returns infinity with warnings instead of raising `ZeroDivisionError`. The current exception-based implementation does not catch NumPy warnings as Python exceptions. Explicit denominator-zero checks and regression tests for such cases are still needed. Passing the current suite therefore does not establish correctness for every supported input.

## 7 Deliverables and contribution

| Requirement in the course brief | Repository evidence |
|---|---|
| Shared implementation with setup and usage documentation | `dual.py`, `README.md`, `requirements.txt` |
| Short demonstration script or notebook | `demo.py` demonstrates a function at one point and runs the verification suite |
| At least three analytical comparisons at multiple points | Section 2 and the printed analytical tables |
| More involved composition with automatic chain-rule explanation | Section 3, including symbolic propagation and a numerical trace |
| Reusable function interface | Section 4 and `evaluate_ad()` |
| Forward finite differences, errors, and cancellation discussion | Section 5 and the 16-step table |
| Report of experiments and observed errors | This report |
| Team contributions | Core representation, arithmetic, elementary functions, and verification roles described below |

The course role allocation places core representation with Member 1, arithmetic with Member 2, elementary functions/domains with Member 3, and verification/demonstrations with Member 4. Member 4's contribution verifies the shared implementation, supplies reusable examples and numerical experiments, and explains the evidence and its limits.

## 8 Interpretation

The tested engine propagates first derivatives correctly for the selected arithmetic and composed-function examples. Analytical references make correctness measurable; the finite-difference experiment demonstrates why choosing a smaller step does not always improve a numerical derivative. The reusable interface separates evaluating a derivative from writing a reference formula for validation. The remaining NumPy-zero issue identifies a specific boundary where error handling needs improvement.

Source requirements: the supplied Scientific Computing Team Project brief, **Project 6 Automatic Differentiation Engine**, file `Project_6_Automatic_Differentiation_Engine.docx`. Implementation and experiment sources: `dual.py` and `testing.py`. The tables were generated from those files. The standalone demonstration is provided in `demo.py`.

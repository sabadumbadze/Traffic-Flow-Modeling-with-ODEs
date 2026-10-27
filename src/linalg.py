"""Small dense linear-algebra helpers (pure Python)."""

from __future__ import annotations

import math
from typing import List, Sequence


class SingularMatrixError(ArithmeticError):
    """Raised when a linear system has a (numerically) singular matrix."""


def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(rhs)
    work = [[float(v) for v in matrix[i]] + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-300:
            raise SingularMatrixError("singular matrix in solve_linear")
        work[col], work[pivot] = work[pivot], work[col]
        for row in range(col + 1, n):
            factor = work[row][col] / work[col][col]
            if factor != 0.0:
                for k in range(col, n + 1):
                    work[row][k] -= factor * work[col][k]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        tail = sum(work[i][j] * solution[j] for j in range(i + 1, n))
        solution[i] = (work[i][n] - tail) / work[i][i]
    return solution


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm_inf(v: Sequence[float]) -> float:
    return max((abs(x) for x in v), default=0.0)


def norm_2(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def is_finite_vector(v: Sequence[float]) -> bool:
    return all(math.isfinite(x) for x in v)

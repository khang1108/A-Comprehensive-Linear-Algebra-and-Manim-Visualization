import copy
import os
import sys
from dataclasses import asdict, dataclass
from math import sqrt
from time import perf_counter
from typing import Dict, List, Optional

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from part1.gaussian import back_substitution, gaussian_eliminate
from Part2.QR_SVD import dot, identity, norm, svd, transpose


# Numerical thresholds used across solvers.
EPS_ZERO = 1e-12
DEFAULT_TOL_ITER = 1e-8
DEFAULT_TOL_QR = 1e-12
DEFAULT_TOL_SVD = 1e-10


@dataclass
class SolverResult:
    """Unified result container for all solvers."""

    method: str        # solver name
    x: List[float]     # solution vector
    success: bool      # True if solved / converged
    residual: float    # ||Ax - b||_2
    iterations: int    # number of iterations (0 for direct methods)
    runtime_sec: float # wall-clock time in seconds
    message: str       # status message


def validate_inputs(A: List[List[float]], b: List[float]) -> tuple[List[List[float]], List[float]]:
    """Validate inputs and return float copies of A and b."""
    if not isinstance(A, list) or not A or not isinstance(A[0], list):
        raise ValueError("A must be a 2-D matrix.")

    n_rows = len(A)
    n_cols = len(A[0])

    if any(len(row) != n_cols for row in A):
        raise ValueError("All rows of A must have the same length.")

    if n_rows != n_cols:
        raise ValueError(f"A must be square. Got {n_rows}x{n_cols}.")

    if len(b) != n_rows:
        raise ValueError(f"Dimension mismatch: A has {n_rows} rows but b has length {len(b)}.")

    A_copy = [[float(v) for v in row] for row in A]
    b_copy = [float(v) for v in b]
    return A_copy, b_copy


def residual_norm(A: List[List[float]], x: List[float], b: List[float]) -> float:
    """Compute ||Ax - b||_2."""
    total = 0.0
    for i, row in enumerate(A):
        diff = sum(row[j] * x[j] for j in range(len(x))) - b[i]
        total += diff * diff
    return sqrt(total)


def matvec(A: List[List[float]], x: List[float]) -> List[float]:
    """Compute matrix-vector product Ax."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def is_strictly_row_diagonally_dominant(A: List[List[float]]) -> bool:
    """Return True if A is strictly row diagonally dominant: |a_ii| > sum_{j!=i} |a_ij|."""
    for i, row in enumerate(A):
        diag = abs(row[i])
        off_diag_sum = sum(abs(v) for v in row) - diag
        if diag <= off_diag_sum:
            return False
    return True


def qr_householder_decompose(A: List[List[float]], tol: float = DEFAULT_TOL_QR) -> tuple[List[List[float]], List[List[float]]]:
    """
    QR decomposition via Householder reflections.
    Reuses dot, norm, identity, transpose from Part2.

    Returns Q (m x m orthogonal), R (m x n upper-triangular).
    """
    m, n = len(A), len(A[0])
    R = copy.deepcopy(A)
    Q = identity(m)

    for k in range(min(m, n)):
        x = [R[i][k] for i in range(k, m)]
        x_norm = norm(x)
        if x_norm < tol:
            continue

        # sign chosen to avoid cancellation
        alpha = -x_norm if x[0] >= 0 else x_norm
        v = x[:]
        v[0] -= alpha
        beta = dot(v, v)
        if beta < tol:
            continue

        # apply reflector to R from the left: R <- H_k R
        for j in range(k, n):
            proj = 2.0 * sum(v[t] * R[k + t][j] for t in range(len(v))) / beta
            for t in range(len(v)):
                R[k + t][j] -= proj * v[t]

        # accumulate reflector into Q from the right: Q <- Q H_k
        for i in range(m):
            proj = 2.0 * sum(Q[i][k + t] * v[t] for t in range(len(v))) / beta
            for t in range(len(v)):
                Q[i][k + t] -= proj * v[t]

    # zero out numerical noise below the diagonal
    for i in range(m):
        for j in range(min(i, n)):
            if abs(R[i][j]) < tol:
                R[i][j] = 0.0

    return Q, R


def solve_gauss(A: List[List[float]], b: List[float]) -> SolverResult:
    """
    Solve Ax = b using Gaussian elimination with pivoting (Part 1).

    Algorithm:
        1. Forward elimination -> upper-triangular system Ux = c.
        2. Back substitution -> solution x.
    """
    start = perf_counter()
    A_mat, b_vec = validate_inputs(A, b)

    U, c, _ = gaussian_eliminate(A_mat, b_vec)
    x, status = back_substitution(U, c)

    runtime = perf_counter() - start

    # In Part1, "Singular Solution" actually means a unique solution.
    if status != "Singular Solution":
        return SolverResult(
            method = "Gauss",
            x = [],
            success = False,
            residual = float("inf"),
            iterations = 0,
            runtime_sec = runtime,
            message = status,
        )

    x = [float(v) for v in x]
    return SolverResult(
        method = "Gauss",
        x = x,
        success = True,
        residual = residual_norm(A_mat, x, b_vec),
        iterations = 0,
        runtime_sec = runtime,
        message = "Solved using Gaussian elimination + back substitution.",
    )


def solve_gauss_seidel(
    A: List[List[float]],
    b: List[float],
    x0: Optional[List[float]] = None,
    tol: float = DEFAULT_TOL_ITER,
    max_iter: int = 1000,
) -> SolverResult:
    """
    Solve Ax = b using the Gauss-Seidel iterative method.

    Iteration formula (at step k, component i):
        x_i^(k+1) = (b_i - sum_{j<i} a_ij x_j^(k+1) - sum_{j>i} a_ij x_j^(k)) / a_ii

    Sufficient convergence condition: A is strictly row diagonally dominant.
    Stopping criterion: ||x^(k+1) - x^(k)||_inf < tol.

    Parameters:
        x0: initial guess, defaults to zero vector.
        tol: convergence tolerance.
        max_iter: maximum number of iterations.
    """
    start = perf_counter()
    A_mat, b_vec = validate_inputs(A, b)
    n = len(A_mat)

    # Gauss-Seidel requires non-zero diagonal entries.
    if any(abs(A_mat[i][i]) < EPS_ZERO for i in range(n)):
        return SolverResult(
            method = "Gauss-Seidel",
            x = [],
            success = False,
            residual = float("inf"),
            iterations = 0,
            runtime_sec = perf_counter() - start,
            message = "Zero (or near-zero) diagonal entry; cannot apply Gauss-Seidel.",
        )

    if x0 is None:
        x = [0.0] * n
    else:
        if len(x0) != n:
            raise ValueError(f"x0 must have length {n}. Received {len(x0)}.")
        x = [float(v) for v in x0]

    # Convergence-condition check (warning only, no hard stop).
    warn = (
        ""
        if is_strictly_row_diagonally_dominant(A_mat)
        else "Matrix is not strictly row diagonally dominant - convergence is not guaranteed. "
    )

    ok = False
    residual = float("inf")

    for k in range(1, max_iter + 1):
        x_old = x[:]

        for i in range(n):
            left = sum(A_mat[i][j] * x[j] for j in range(i))
            right = sum(A_mat[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b_vec[i] - left - right) / A_mat[i][i]

        residual = residual_norm(A_mat, x, b_vec)
        max_delta = max(abs(x[i] - x_old[i]) for i in range(n))

        if max_delta < tol:
            ok = True
            iterations = k
            break
    else:
        iterations = max_iter

    runtime = perf_counter() - start
    status = "Converged." if ok else "Reached max_iter without convergence."
    return SolverResult(
        method = "Gauss-Seidel",
        x = x,
        success = ok,
        residual = residual,
        iterations = iterations,
        runtime_sec = runtime,
        message = warn + status,
    )


def solve_qr_householder(
    A: List[List[float]], b: List[float], tol: float = DEFAULT_TOL_QR
) -> SolverResult:
    """
    Solve Ax = b using QR-Householder decomposition.

    Algorithm:
        1. Factorize A = QR using Householder reflections.
        2. Compute y = Q^T b.
        3. Solve Rx = y using back substitution (Part 1).

    Utility functions from Part2 (dot, norm, identity, transpose)
    are reused in the QR factorization step.
    """
    start = perf_counter()
    A_mat, b_vec = validate_inputs(A, b)

    try:
        Q, R_full = qr_householder_decompose(A_mat, tol = tol)
        Qt = transpose(Q)
        y_full = matvec(Qt, b_vec)

        n = len(A_mat)
        R = [R_full[i][:n] for i in range(n)]
        y = y_full[:n]

        x, status = back_substitution(R, y)
        if status != "Singular Solution":
            raise ValueError(status)

        x = [float(v) for v in x]
        return SolverResult(
            method = "QR-Householder",
            x = x,
            success = True,
            residual = residual_norm(A_mat, x, b_vec),
            iterations = 0,
            runtime_sec = perf_counter() - start,
            message = "Solved using QR-Householder decomposition.",
        )
    except Exception as exc:
        return SolverResult(
            method = "QR-Householder",
            x = [],
            success = False,
            residual = float("inf"),
            iterations = 0,
            runtime_sec = perf_counter() - start,
            message = f"QR-Householder failed: {exc}",
        )


def solve_svd(
    A: List[List[float]], b: List[float], tol: float = DEFAULT_TOL_SVD
) -> SolverResult:
    """
    Solve Ax = b using the Moore-Penrose pseudo-inverse via SVD (Part 2).

    Algorithm:
        1. Decompose A = U Σ V^T using svd() from Part 2.
        2. Compute x = V Σ^+ U^T b, where Σ^+ ignores singular values <= tol.

    For non-singular square systems, this matches the exact solution.
    """
    start = perf_counter()
    A_mat, b_vec = validate_inputs(A, b)

    try:
        U, S, Vt = svd(A_mat)
        m, n = len(A_mat), len(A_mat[0])

        V = [[Vt[j][i] for j in range(len(Vt))] for i in range(len(Vt[0]))]
        sigma = [S[i][i] if i < m and i < n else 0.0 for i in range(n)]

        # y = U^T b
        y = [sum(U[r][i] * b_vec[r] for r in range(m)) for i in range(n)]

        # z = Σ^+ y
        z = [0.0] * n
        rank = 0
        for i in range(n):
            if sigma[i] > tol:
                z[i] = y[i] / sigma[i]
            rank += 1

        # x = V z
        x = [sum(V[i][j] * z[j] for j in range(n)) for i in range(n)]
        return SolverResult(
            method = "SVD",
            x = x,
            success = True,
            residual = residual_norm(A_mat, x, b_vec),
            iterations = 0,
            runtime_sec = perf_counter() - start,
            message = f"Solved using Part2 SVD (effective rank = {rank}).",
        )
    except Exception as exc:
        return SolverResult(
            method = "SVD",
            x = [],
            success = False,
            residual = float("inf"),
            iterations = 0,
            runtime_sec = perf_counter() - start,
            message = f"SVD failed: {exc}",
        )


def run_all_solvers(
    A: List[List[float]],
    b: List[float],
    x0: Optional[List[float]] = None,
    tol_seidel: float = DEFAULT_TOL_ITER,
    max_iter_seidel: int = 1000,
    tol_qr: float = DEFAULT_TOL_QR,
    tol_svd: float = DEFAULT_TOL_SVD,
) -> Dict[str, Dict[str, object]]:
    """Run all 4 methods on the same system and return a notebook-friendly dictionary."""
    return {
        "gauss": asdict(solve_gauss(A, b)),
        "gauss_seidel": asdict(solve_gauss_seidel(A, b, x0 = x0, tol = tol_seidel, max_iter = max_iter_seidel)),
        "qr_householder": asdict(solve_qr_householder(A, b, tol = tol_qr)),
        "svd": asdict(solve_svd(A, b, tol = tol_svd)),
    }


if __name__ == "__main__":
    A_test = [
        [4.0, 1.0, 2.0],
        [3.0, 5.0, 1.0],
        [1.0, 1.0, 3.0],
    ]
    b_test = [4.0, 7.0, 3.0]

    results = run_all_solvers(A_test, b_test, tol_seidel = 1e-10, max_iter_seidel = 500)
    for name, result in results.items():
        print(f"{name}: {result}")

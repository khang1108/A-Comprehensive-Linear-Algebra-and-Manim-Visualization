from typing import List
from gaussian import gaussian_eliminate

import copy

def determinant(A: List[List[float]]) -> float:
    """
    Compute det(A) using Gaussian Elimination with Partial Pivoting.

    Algorithm:
        Step 1: Validate that A is a square matrix (n × n).
        Step 2: Create a copy of A to preserve the original data.
        Step 3: Apply Gaussian Elimination with partial pivoting
                 to reduce A to upper triangular form U.
                 - At each step k, select the row with the largest
                   |element| in column k (from row k downward) as pivot.
                 - Each row swap flips the sign of the determinant.
                 - Eliminate elements below the pivot via row subtraction.
        Step 4: Compute det(A) using the formula (Equation 3):
                 det(A) = (-1)^s × ∏(i=1→n) u_ii
                 where s = number of row swaps, u_ii = diagonal of U.

    Parameters:
        A (List[List[float]]): A square n × n matrix (numpy array or list of lists).

    Returns:
        float: The determinant det(A). Returns 0.0 if A is singular.

    Raises:
        ValueError: If A is not a square matrix.
    """

    A_copy = copy.deepcopy(A)
    n_rows, n_cols = len(A), len(A[0])

    if n_rows != n_cols:
        raise ValueError(
            f"Matrix must be square to compute determinant. Got {n_rows}×{n_cols}."
        )

    n = n_rows

    U, _, n_swaps = gaussian_eliminate(A_copy, only_one=True)

    det = 1.0
    for i in range(n):
        det *= U[i][i]
    det *= (-1) ** n_swaps

    return det
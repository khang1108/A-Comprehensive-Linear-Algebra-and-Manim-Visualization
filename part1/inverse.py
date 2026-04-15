from determinant import determinant
from gaussian import gaussian_eliminate, back_substitution
from typing import Any, List

import copy

# TODO: inverse(A)
def inverse(A: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(A[0])

    # Check if the matrix is square
    if n != m:
        raise ValueError("The matrix must be squared")
    if determinant(A) == 0.0:
        raise ValueError("Determinant equals to 0")

    # Initialize the inverse matrix
    inv = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Create the unit vector e_i
        e = [0.0] * n
        e[i] = 1.0

        # Copy A because gaussian_eliminate may modify it
        A_copy = copy.deepcopy(A)

        # Apply Gaussian elimination to the augmented system (A | e_i)
        mat = gaussian_eliminate(A_copy, e)

        # Convert result to list and separate U and c
        U, c, _ = mat

        # Solve Ux = c using back substitution
        x, _ = back_substitution(U, c)

        # Assign solution as the i-th column of the inverse matrix
        for j in range(n):
            inv[j][i] = x[j]

    return inv
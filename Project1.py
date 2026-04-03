from dataclasses import dataclass
from typing import Any, List

import numpy as np

def gaussian_eliminate(A : List[List[float]], b : List[float]) -> List[float]:
    """
    Reducing rows in a matrix by Gaussian Eliminate Algortihm.

    Parameters:
        - A (List[List[float]]): A matrix m x n
        - b (List[float]): A vector

    Returns:
        A: matrix after Gaussian Eliminate
    """

    # (A|B) = A.concat(B, axis=1)
    # In this step, I reshape Vector B into a column vector 1 x m
    # Then I concatenate A and B to form a new matrix (A|B)
    b = b.reshape(-1, 1)
    mat = np.concatenate((A, b), axis=1)

    n_rows, n_cols = mat.shape

    '''
    Partial Gaussian Elimination:
        - Create matrix (A|b)
        - Find the row with LARGEST ABSOLUTE VALUE in the first COLUMN and SWAP 
          that row with the first row
        - Divide the first row by the pivot element to make it equal to 1
        - Use the first row to eliminate the first column in all other rows by 
        subtracting the appropriate multiple of the first row from each of the 
        - Repeate step 2-4 for the remaining columns, using the row with the 
          largest absolute value in the current column as the pivot row
        - Back-substitue to obtain the values of the unknowns
    '''

    for i in range(n_rows):
        pivot_row = i
        
        for j in range(i + 1, n_rows):
            if abs(mat[j, i]) > abs(mat[pivot_row, i]):
                pivot_row = j
        # Swap the pivot row with the first row 
        mat[[i, pivot_row]] = mat[[pivot_row, i]]

        for k in range(i + 1, n_rows):
            if mat[i, i] == 0:
                # Skip the row if the pivot element is 0
                continue
            mul = mat[k, i] / mat[i, i]

            # Update i-th row to eliminate the i-th column
            mat[k, i:] = mat[k, i:] - mul * mat[i, i:]
    
    return mat

#TODO: Implement back_substitution(U, c)
def back_substitution(U: List[List[float]], c: List[float]) -> Any:
    pass

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

    # Step 1: Input validation — must be square
    A = np.array(A, dtype=float)
    n_rows, n_cols = A.shape

    if n_rows != n_cols:
        raise ValueError(
            f"Matrix must be square to compute determinant. "
            f"Got {n_rows}×{n_cols}."
        )

    n = n_rows

    # Step 2: Copy to avoid mutating the original
    mat = A.copy()

    # Step 3: Gaussian Elimination with Partial Pivoting
    s = 0  # Row swap counter (each swap flips det sign)

    for k in range(n):
        # Step 3a: Find pivot — row with max |element| in column k
        pivot_row = k
        for i in range(k + 1, n):
            if abs(mat[i, k]) > abs(mat[pivot_row, k]):
                pivot_row = i

        # Step 3b: Singular check — pivot is zero
        if abs(mat[pivot_row, k]) == 0:
            return 0.0

        # Step 3c: Swap rows if needed (flips det sign)
        if pivot_row != k:
            mat[[k, pivot_row]] = mat[[pivot_row, k]]
            s += 1

        # Step 3d: Eliminate entries below pivot
        for i in range(k + 1, n):
            factor = mat[i, k] / mat[k, k]
            mat[i, k:] = mat[i, k:] - factor * mat[k, k:]  # R_i ← R_i - factor × R_k

    # Step 4: det(A) = (-1)^s × product of diagonal entries of U
    diagonal_product = 1.0
    for i in range(n):
        diagonal_product *= mat[i, i]

    det_value = ((-1) ** s) * diagonal_product
    return det_value

#TODO: inverse(A)
def inverse(A: List[List[float]]) -> List[List[float]]:
    pass

#TODO: rank_and_basis(A)
def rank_and_basis(A: List[List[float]]) -> Any:
    pass

#TODO: verify_solution(A, x, b)


if __name__ == "__main__":
    A = np.array([
        [2.0, 1.0],
        [4.0, -6.0]
    ])

    b = np.array([5.0, -2.0])

    print(gaussian_eliminate(A, b))

    # Test determinant
    # Test 1: 2x2 basic
    B = np.array([[2.0,  1.0],
                  [4.0, -6.0]])
    print(determinant(B))

    # Test 2: 3x3 singular — linearly dependent rows
    C = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])
    print(determinant(C))

    # Test 3: 2x2 — pivot swap needed since a[0,0] = 0
    D = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    print(determinant(D))
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

#TODO: Implement determinant(A)
def determinant(A: List[List[float]]) -> int:
    pass

#TODO: inverse(A)
def inverse(A: List[List[float]]) -> List[List[float]]:
    n = len(A)

    # 1. Check if the matrix is square
    for row in A:
        if len(row) != n:
            raise ValueError("Matrix must be square (n x n).")

    # 2. Check if the matrix is singular (non-invertible)
    # Using numpy's determinant check for robustness
    if np.isclose(np.linalg.det(np.array(A)), 0):
        raise ValueError("Matrix is singular and cannot be inverted (determinant is zero).")

    # 3. Initialize the inverse matrix with zeros
    inv = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Create the i-th unit vector e_i
        e = [0.0] * n
        e[i] = 1.0

        # Copy A to avoid modifying the original data during Gaussian elimination
        A_copy = [row[:] for row in A]

        try:
            # 4. Apply Gaussian elimination to the augmented system (A | e_i)
            # Ensure input to your function is consistent (numpy arrays)
            mat = gaussian_eliminate(np.array(A_copy, dtype=float), 
                                     np.array(e, dtype=float))

            # 5. Separate U and c from the result [U | c]
            mat_list = mat.tolist()
            U = [row[:n] for row in mat_list]   # Upper triangular part
            c = [row[n] for row in mat_list]    # Transformed right-hand side

            # 6. Solve Ux = c using back substitution
            x = back_substitution(U, c)

            # 7. Assign the solution x as the i-th column of the inverse matrix
            for j in range(n):
                inv[j][i] = x[j]
        
        except ZeroDivisionError:
            raise ValueError("Zero pivot encountered during Gaussian elimination. Matrix is not invertible.")

    return inv

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
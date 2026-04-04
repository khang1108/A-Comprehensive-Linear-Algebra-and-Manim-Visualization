from typing import Any, List

import numpy as np


def gaussian_eliminate(A: np.ndarray, b: np.ndarray):
    """
    Reducing rows in a matrix by Gaussian Eliminate Algortihm.

    Parameters:
        - A (List[List[float]]): A matrix m x n
        - b (List[float]): A vector

    Returns:
        U: Upper triangular matrix after Gaussian Eliminate
        c: Transformed vector b
    """

    # (A|B) = A.concat(B, axis=1)
    # In this step, I reshape Vector B into a column vector 1 x m
    # Then I concatenate A and B to form a new matrix (A|B)
    b = b.reshape(-1, 1)
    mat = np.concatenate((A, b), axis=1)

    n_rows = mat.shape[0]

    """
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
    """

    for i in range(n_rows):
        pivot_row = i

        for j in range(i + 1, n_rows):
            if abs(mat[j, i]) > abs(mat[pivot_row, i]):
                pivot_row = j
        # Swap the pivot row with the first row
        mat[[i, pivot_row]] = mat[[pivot_row, i]]

        if mat[i, i] == 0:
            # Skip the row if the pivot element is 0
            continue

        for k in range(i + 1, n_rows):
            mul = mat[k, i] / mat[i, i]

            # Update i-th row to eliminate the i-th column
            mat[k, i:] = mat[k, i:] - mul * mat[i, i:]

    return mat[:, :-1], mat[:, -1]


# TODO: Implement back_substitution(U, c)
def back_substitution(U: np.ndarray, c: np.ndarray) -> Any:
    """
    Solving the linear system Ux = c by Back Substitution Algorithm.
    Handles unique solution, no solution, and infinitely many solutions.

    Parameters:
        - U (np.ndarray): An upper triangular matrix m x n
        - c (np.ndarray): A vector

    Returns:
        - List[float]: If there is a unique solution.
        - str: If there is no solution, or a general formula for infinite solutions.
    """

    # In this step, I use an (n x n+1) matrix to store the general solution formula.
    # Column 0 stores the constant term.
    # Column 1 to n stores the coefficients of the free variables x_0 to x_{n-1}.
    n = U.shape[0]

    x = np.zeros(n, dtype=float)
    """
    Back Substitution with General Solution Handling:
        - Iterate BACKWARDS from the LAST row down to the FIRST row
        - If the pivot is zero, CHECK consistency to identify 'No Solution' or 'Infinite Solutions'
        - If it has infinite solutions, MARK the current variable as a FREE VARIABLE
        - Otherwise, CALCULATE the symbolic expression using array arithmetic
    """

    for i in range(n - 1, -1, -1):
        # Calculate sum of U[i][j] * x_expr[j] for known variables
        sum_ux = 0.0

        for j in range(i + 1, n):
            sum_ux += U[i, j] * x[j]

        x[i] = (c[i] - sum_ux) / U[i, i]

    return x


# TODO: Implement determinant(A)
def determinant(A: List[List[float]]):
    pass


# TODO: inverse(A)
def inverse(A: List[List[float]]):
    pass


# TODO: rank_and_basis(A)
def rank_and_basis(A: List[List[float]]) -> Any:
    pass


# TODO: verify_solution(A, x, b)


if __name__ == "__main__":
    A = np.array([[2.0, 1.0], [4.0, -6.0]])

    b = np.array([5.0, -2.0])

    U, c = gaussian_eliminate(A, b)
    print(back_substitution(U, c))


from typing import List
import copy

def gaussian_eliminate(
    A: List[List[float]], b: List[float] = [], only_one: bool = False
):
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
    # In this step, I reshape Vector B into a column vector m x 1
    # Then I concatenate A and B to form a new matrix (A|B)
    n_rows, n_cols = len(A), len(A[0])
    if not only_one:
        b_new = []
        for x in b:
            b_new.append(x)
        b = b_new
        n_cols += 1

        mat = []
        for i in range(n_rows):
            new_row = A[i] + [b[i]]
            mat.append(new_row)
    else:
        mat = copy.deepcopy(A)

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

    n_swap = 0
    for i in range(min(n_rows, n_cols)):
        pivot_row = i

        for j in range(i + 1, n_rows):
            if abs(mat[j][i]) > abs(mat[pivot_row][i]):
                pivot_row = j
        # Swap the pivot row with the first row
        if i != pivot_row:
            mat[i], mat[pivot_row] = mat[pivot_row], mat[i]
            n_swap += 1

        if abs(mat[i][i]) < 1e-12:
            # Skip the row if the pivot element is 0
            continue

        for j in range(i + 1, n_rows):
            mul = mat[j][i] / mat[i][i]

            # Update i-th row to eliminate the i-th column
            for k in range(i, n_cols):
                mat[j][k] -= mul * mat[i][k]

    if only_one:
        U = mat
        c = []
    else:
        U = [row[: n_cols - 1] for row in mat]
        c = [row[n_cols - 1] for row in mat]

    return U, c, n_swap

# TODO: Implement back_substitution(U, c)
def back_substitution(U: List[List[float]], c: List[float]) -> Any:
    """
    Solving the linear system Ux = c by Back Substitution Algorithm.
    Handles unique solution, no solution, and infinitely many solutions.

    Parameters:
        - U (List[List[float]]): An upper triangular matrix m x n
        - c (List[float]): A vector

    Returns:
        - List[float]: If there is a unique solution.
        - str: If there is no solution, or a general formula for infinite solutions.
    """

    # In this step, I use an (n x n+1) matrix to store the general solution formula.
    # Column 0 stores the constant term.
    # Column 1 to n stores the coefficients of the free variables x_0 to x_{n-1}.
    n_rows, n_cols = len(U), len(U[0])

    x = [0.0] * n_cols
    """
    Back Substitution with General Solution Handling:
        - Iterate BACKWARDS from the LAST row down to the FIRST row
        - If the pivot is zero, CHECK consistency to identify 'No Solution' or 'Infinite Solutions'
        - If it has infinite solutions, MARK the current variable as a FREE VARIABLE
        - Otherwise, CALCULATE the symbolic expression using array arithmetic
    """

    for i in range(n_rows - 1, -1, -1):
        # Calculate sum of U[i][j] * x_expr[j] for known variables
        sum = 0.0

        for j in range(i + 1, n_cols):
            sum += U[i][j] * x[j]

        if abs(U[i][i]) < 1e-12:
            if abs(c[i] - sum) > 1e-12:
                return [], "No Solution"
            else:
                return [], "Infinitively many solutions"

        x[i] = (c[i] - sum) / U[i][i]

    return x, "Singular Solution"
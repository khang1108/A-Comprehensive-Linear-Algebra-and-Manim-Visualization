from typing import Any, List
from scipy import linalg as la

import copy
import numpy as np


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


# TODO: rank_and_basis(A)
def transpose(A: List[List[float]]) -> List[List[float]]:
    n_rows = len(A)
    n_cols = len(A[0])
    return [[A[j][i] for j in range(n_rows)] for i in range(n_cols)]


def rank_and_basis(A: List[List[float]]) -> Any:
    U, _, _ = gaussian_eliminate(A, only_one=True)
    rank = 0

    n_rows, n_cols = len(U), len(U[0])
    for i in range(n_rows):
        is_zero = True
        for j in range(n_cols):
            if U[i][j] != 0.0:
                is_zero = False
                break

        if is_zero:
            rank += 1

    row_basis = U[:rank]

    AT = transpose(A)
    UT, _, _ = gaussian_eliminate(AT, only_one=True)

    col_basis = UT[:rank]

    return {"rank": rank, "row_basis": row_basis, "col_basis": col_basis}


# TODO: verify_solution(A, x, b)


if __name__ == "__main__":
    A = [[2.0, 1.0], [4.0, -6.0]]
    b = [5.0, -2.0]
    # 1. Chạy code của bạn
    U, c, n_swaps = gaussian_eliminate(A, b)
    x_sol, message = back_substitution(U, c)

    print("-" * 30)
    print(f"My result: {x_sol}")
    print(f"Message: {message}")
    print(f"Swaps: {n_swaps}")

    # 2. Thử nghiệm với thư viện (Sử dụng try-except vì ma trận suy biến)
    try:
        x_correct = la.solve(np.array(A), np.array(b))
        print(f"Library solution: {x_correct}")

        # Kiểm tra Ax = b cho nghiệm duy nhất
        b_check = np.dot(np.array(A), np.array(x_sol))
        is_correct = np.allclose(b_check, np.array(b), atol=1e-10)
        print(f"Status: {'Correct' if is_correct else 'Incorrect'}")

    except la.LinAlgError:
        print("Library Result: Matrix is singular (No unique solution)")

        # Kiểm tra logic của bạn có nhận diện đúng không
        # Với case [1,1],[2,2] và b=[2,5] -> Phải là No Solution
        if message == "No Solution":
            print("Correctly identified No Solution")
        elif message == "Infinitely many solutions":
            print("Wrong! This system is inconsistent (No Solution)")
        else:
            print("Failed to identify singular matrix")

    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        det_manually = determinant(A)

        det_correct = la.det(np.array(A))

        print(f"My result: {det_manually}")
        print(f"Correct solution: {det_correct}")
    except ValueError:
        print("Matrix must be squared matrix")

    try:
        inv = inverse(A)
        print(f"My Result: {inv}")

        print(f"Correct Result: {la.inv(np.array(A))}")
    except ValueError:
        print("Some errors occurred")

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
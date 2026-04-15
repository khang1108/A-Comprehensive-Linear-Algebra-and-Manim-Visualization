from gaussian import gaussian_eliminate
from typing import List, Any

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

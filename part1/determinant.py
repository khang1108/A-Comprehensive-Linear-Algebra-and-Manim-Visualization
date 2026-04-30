"""
Tính định thức ma trận vuông bằng khử Gauss và chọn cột chủ.
"""
from typing import List
from gaussian import gaussian_eliminate

import copy


def _check_matrix(A: List[List[float]]) -> tuple[int, int]:
    """Kiểm tra A khác rỗng, mọi hàng cùng số cột.
    
    Returns:
        (số_hàng, số_cột).
    """
    if not A:
        raise ValueError("Ma trận không được rỗng.")

    n_rows = len(A)
    widths = {len(row) for row in A} # Lưu vào một set để loại bỏ duplicates
    # Dùng để kiểm tra xem mọi hàng có cùng số cột không
    if len(widths) != 1:
        raise ValueError(
            "Mọi hàng của ma trận phải có cùng số cột. "
            f"Các độ rộng gặp được: {sorted(widths)}."
        )
    n_cols = len(A[0])
    return n_rows, n_cols


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

    Trả về:
        Giá trị định thức (float). Ma trận suy biến (định thức đúng bằng 0)
        thì kết quả có thể là 0 hoặc rất gần 0 do sai số dấu phẩy động.

    Raises:
        ValueError: If A is not a square matrix.
    """
    n_rows, n_cols = _check_matrix(A)

    if n_rows != n_cols:
        raise ValueError(
            "Chỉ tính định thức cho ma trận vuông. "
            f"Nhận được kích thước {n_rows} x {n_cols}."
        )

    n = n_rows
    A_copy = copy.deepcopy(A)

    U, _, n_swaps = gaussian_eliminate(A_copy, only_one=True)

    det = 1.0
    for i in range(n):
        det *= U[i][i]
    det *= (-1) ** n_swaps

    return det

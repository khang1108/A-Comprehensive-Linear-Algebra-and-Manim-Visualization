"""
Hạng ma trận và cơ sở cho một số không gian con tiêu chuẩn (hàng, cột, right_null_space, left_null_space).
"""

from typing import Any, Dict, List, Tuple
from gaussian import _EPS, gaussian_eliminate
from determinant import _check_matrix
from determinant import _check_matrix

import copy

def transpose(A: List[List[float]]) -> List[List[float]]:
    """Chuyển vị ma trận A."""
    n_rows = len(A)
    n_cols = len(A[0])
    return [[A[j][i] for j in range(n_rows)] for i in range(n_cols)]


def _row_close_to_zero(row: List[float], eps: float = _EPS) -> bool:
    """True nếu mọi phần tử của hàng có trị tuyệt đối <= eps."""
    return all(abs(x) <= eps for x in row)


def _pivot_info_from_echelon(U: List[List[float]], eps: float = _EPS) -> List[Tuple[int, int]]:
    """
    Lấy thông tin pivot từ ma trận bậc thang U (sau khử Gauss).

    Trả về danh sách (row_idx, pivot_col) theo thứ tự từ trên xuống.
    """
    pivots: List[Tuple[int, int]] = []
    for i, row in enumerate(U):
        pivot_col = None
        for j, val in enumerate(row):
            if abs(val) > eps:
                pivot_col = j
                break
        if pivot_col is not None:
            pivots.append((i, pivot_col))
    return pivots


def _null_basis_from_echelon(
    U: List[List[float]], pivots: List[Tuple[int, int]], n: int
) -> List[List[float]]:
    """
    Tính cơ sở không gian hạch từ hệ Ux=0, với U là dạng bậc thang của A.

    Với mỗi biến tự do f, đặt x_f=1 (các biến tự do khác = 0) rồi thế ngược để tìm biến pivot.
    Dùng khi chỉ có rank(A) < n. Vì nếu rank(A) = n thì không gian nghiệm là {0}.
    """
    pivot_cols = [p for (_, p) in pivots]
    rank = len(pivots)
    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]
    if not free_cols:
        return []

    basis: List[List[float]] = []
    for f in free_cols:
        x = [0.0] * n
        x[f] = 1.0

        for k in range(rank - 1, -1, -1):
            row_idx, p = pivots[k]
            s = 0.0
            for j in range(p + 1, n):
                s += U[row_idx][j] * x[j]
            if abs(U[row_idx][p]) < _EPS:
                x[p] = 0.0
            else:
                x[p] = -s / U[row_idx][p]

        basis.append(x)
    return basis


def rank_and_basis(A: List[List[float]]) -> Dict[str, Any]:
    """
    Tính hạng và cơ sở cho:
        - Row(A): không gian dòng (vector trong R^n)
        - Col(A): không gian cột (vector trong R^m)
        - Nul(A): không gian nghiệm của Ax=0 (vector trong R^n)
    """
    m, n = _check_matrix(A)

    A_copy = copy.deepcopy(A)
    U, _, _ = gaussian_eliminate(A_copy, only_one=True)

    # Rank và pivot
    pivots = _pivot_info_from_echelon(U)
    pivot_cols = [p for (_, p) in pivots]
    rank = len(pivots)

    # Cơ sở không gian hàng: các hàng khác (gần) không của U
    row_basis = [list(row) for row in U if not _row_close_to_zero(row)]

    # Cơ sở không gian cột: lấy các cột pivot của A gốc
    col_basis: List[List[float]] = []
    for j in pivot_cols:
        col_basis.append([float(A[i][j]) for i in range(m)])

    # Không gian nghiệm: nghiệm của Ux = 0
    null_basis = _null_basis_from_echelon(U, pivots, n)

    return {
        "rank": rank,
        "row_basis": row_basis,
        "col_basis": col_basis,
        "null_basis": null_basis,
    }

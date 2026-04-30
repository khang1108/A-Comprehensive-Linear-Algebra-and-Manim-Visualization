from determinant import determinant, _check_matrix
from gaussian import gaussian_eliminate, back_substitution, _EPS
from typing import Any, List

import copy


def inverse(A: List[List[float]]) -> List[List[float]]:
    """
    Tính ma trận nghịch đảo A^{-1} (phương pháp tương đương Gauss–Jordan theo từng cột).

    Với mỗi cột i của I_n, giải hệ A x = e_i (e_i là vector đơn vị thứ i),
    nghiệm x là cột thứ i của A^{-1}.

    Tham số:
        A: Ma trận vuông khả nghịch, kích thước n x n.

    Trả về:
        Ma trận n x n là A^{-1}.

    Ngoại lệ:
        ValueError: Ma trận không vuông, không hợp lệ, suy biến,
        hoặc không giải được đủ n hệ với nghiệm duy nhất.
    """
    n_rows, n_cols = _check_matrix(A)

    if n_rows != n_cols:
        raise ValueError(
            "Chỉ tính nghịch đảo cho ma trận vuông. "
            f"Nhận được kích thước {n_rows} x {n_cols}."
        )

    n = n_rows

    # Định thức gần 0 coi là suy biến (tránh so sánh tuyệt đối == 0.0)
    det_a = determinant(A)
    if abs(det_a) < _EPS:
        raise ValueError(
            "Ma trận suy biến hoặc gần suy biến (định thức gần 0), không có nghịch đảo ổn định."
        )

    inv: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0

        A_copy = copy.deepcopy(A)
        U, c, _ = gaussian_eliminate(A_copy, e)
        x, status = back_substitution(U, c)

        if status != "Hệ có nghiệm duy nhất":
            raise ValueError(
                f"Không thể tính cột {i + 1} của ma trận nghịch đảo: {status}"
            )

        for j in range(n):
            inv[j][i] = x[j]

    return inv

from typing import Any, List, Dict
import numpy as np
import scipy.linalg as la

# from decomposition import qr_decomposition, svd_decomposition

def verify_qr(
    A: List[List[float]], Q: List[List[float]], R: List[List[float]], atol: float = 1e-10
) -> Dict[str, bool]:
    """
    Verify the QR decomposition result: A = QR.

    Parameters:
        - A (List[List[float]]): Original matrix m x n
        - Q (List[List[float]]): Orthogonal matrix m x m (or m x n for reduced)
        - R (List[List[float]]): Upper triangular matrix m x n (or n x n for reduced)
        - atol (float): Absolute tolerance for numerical comparison

    Returns:
        Dict[str, bool]: Verification results for orthogonality, upper triangular form, and reconstruction.
    """
    A_np = np.array(A)
    Q_np = np.array(Q)
    R_np = np.array(R)

    # 1. Check if Q is orthogonal/orthonormal (Q^T * Q ≈ I)
    n_cols_Q = Q_np.shape[1]
    I_expected = np.eye(n_cols_Q)
    is_Q_orthogonal = np.allclose(Q_np.T @ Q_np, I_expected, atol=atol)

    # 2. Check if R is upper triangular
    # np.triu returns the upper triangle, if R is already upper triangular, they should be equal
    is_R_upper_triangular = np.allclose(R_np, np.triu(R_np), atol=atol)

    # 3. Check if A = Q * R
    is_reconstructed = np.allclose(A_np, Q_np @ R_np, atol=atol)

    return {
        "Q_is_orthogonal": is_Q_orthogonal,
        "R_is_upper_triangular": is_R_upper_triangular,
        "A_equals_QR": is_reconstructed,
        "is_correct": is_Q_orthogonal and is_R_upper_triangular and is_reconstructed
    }


def verify_svd(
    A: List[List[float]], U: List[List[float]], S: List[List[float]], V_T: List[List[float]], atol: float = 1e-10
) -> Dict[str, bool]:
    """
    Verify the SVD decomposition result: A = U * S * V^T.

    Parameters:
        - A (List[List[float]]): Original matrix m x n
        - U (List[List[float]]): Orthogonal matrix m x m (left singular vectors)
        - S (List[List[float]]): Diagonal matrix m x n (singular values)
        - V_T (List[List[float]]): Orthogonal matrix n x n (right singular vectors transposed)
        - atol (float): Absolute tolerance for numerical comparison

    Returns:
        Dict[str, bool]: Verification results for orthogonality, diagonal form, and reconstruction.
    """
    A_np = np.array(A)
    U_np = np.array(U)
    S_np = np.array(S)
    V_T_np = np.array(V_T)

    # 1. Check if U is orthogonal (U^T * U ≈ I)
    n_cols_U = U_np.shape[1]
    is_U_orthogonal = np.allclose(U_np.T @ U_np, np.eye(n_cols_U), atol=atol)

    # 2. Check if V_T is orthogonal (V_T * V_T^T ≈ I)
    n_rows_V_T = V_T_np.shape[0]
    is_V_orthogonal = np.allclose(V_T_np @ V_T_np.T, np.eye(n_rows_V_T), atol=atol)

    # 3. Check if S is diagonal (all off-diagonal elements are 0)
    # Create a zero matrix of the same shape as S, then fill its diagonal with S's diagonal
    S_diag_only = np.zeros_like(S_np)
    np.fill_diagonal(S_diag_only, np.diagonal(S_np))
    is_S_diagonal = np.allclose(S_np, S_diag_only, atol=atol)

    # 4. Check if A = U * S * V^T
    is_reconstructed = np.allclose(A_np, U_np @ S_np @ V_T_np, atol=atol)

    return {
        "U_is_orthogonal": is_U_orthogonal,
        "V_is_orthogonal": is_V_orthogonal,
        "S_is_diagonal": is_S_diagonal,
        "A_equals_USV": is_reconstructed,
        "is_correct": is_U_orthogonal and is_V_orthogonal and is_S_diagonal and is_reconstructed
    }


if __name__ == "__main__":
    # Test case 1: Ma trận vuông
    A_square = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
    A_np = np.array(A_square)

    print("=" * 40)
    print("KIỂM CHỨNG PHÂN RÃ QR (Sử dụng Mock Data)")
    print("=" * 40)
    
    # Giả lập kết quả từ hàm do thành viên khác viết (tạm dùng thư viện)
    Q_mock, R_mock = np.linalg.qr(A_np)
    
    # 1. Chạy code kiểm chứng của bạn
    qr_status = verify_qr(A_square, Q_mock.tolist(), R_mock.tolist())
    
    print("-" * 30)
    print(f"Q is orthogonal: {'Pass' if qr_status['Q_is_orthogonal'] else 'Fail'}")
    print(f"R is upper triangular: {'Pass' if qr_status['R_is_upper_triangular'] else 'Fail'}")
    print(f"A = QR: {'Pass' if qr_status['A_equals_QR'] else 'Fail'}")
    print(f"Status: {'Correct' if qr_status['is_correct'] else 'Incorrect'}")

    print("\n" + "=" * 40)
    print("KIỂM CHỨNG PHÂN RÃ SVD (Sử dụng Mock Data)")
    print("=" * 40)

    # Giả lập kết quả từ hàm do thành viên khác viết
    # Lưu ý: np.linalg.svd trả về vector 1D cho S, ta cần chuyển thành ma trận đường chéo
    # để khớp với định nghĩa lý thuyết (A = U * Sigma * V^T)
    U_mock, S_vec, V_T_mock = np.linalg.svd(A_np)
    S_mock = np.zeros_like(A_np, dtype=float)
    np.fill_diagonal(S_mock, S_vec)

    # 2. Chạy code kiểm chứng của bạn
    svd_status = verify_svd(A_square, U_mock.tolist(), S_mock.tolist(), V_T_mock.tolist())

    print("-" * 30)
    print(f"U is orthogonal: {'Pass' if svd_status['U_is_orthogonal'] else 'Fail'}")
    print(f"V is orthogonal: {'Pass' if svd_status['V_is_orthogonal'] else 'Fail'}")
    print(f"S is diagonal: {'Pass' if svd_status['S_is_diagonal'] else 'Fail'}")
    print(f"A = U*S*V^T: {'Pass' if svd_status['A_equals_USV'] else 'Fail'}")
    print(f"Status: {'Correct' if svd_status['is_correct'] else 'Incorrect'}")
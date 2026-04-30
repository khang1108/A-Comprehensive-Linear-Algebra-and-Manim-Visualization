import math

# ================= BASIC =================

def transpose(A):
    """
    Tìm ma trận chuyển vị của A.
    
    Tham số:
        A: Ma trận gốc kích thước m x n.
    Trả về:
        Ma trận chuyển vị kích thước n x m.
    """
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def dot(u, v):
    """
    Tính tích vô hướng (dot product) của hai vector u và v.
    
    Tham số:
        u, v: Hai vector cùng kích thước n.
    Trả về:
        Tích vô hướng (float).
    """
    return sum(x*y for x, y in zip(u, v))

def norm(v):
    """
    Tính chuẩn bậc 2 (L2 norm / Euclidean norm) của vector v.
    
    Tham số:
        v: Vector đầu vào.
    Trả về:
        Độ dài của vector (float).
    """
    return math.sqrt(dot(v, v))

def matmul(A, B):
    """
    Nhân hai ma trận A và B.
    
    Tham số:
        A: Ma trận kích thước m x k.
        B: Ma trận kích thước k x n.
    Trả về:
        Ma trận kết quả kích thước m x n.
    """
    return [[sum(A[i][k]*B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))]
             for i in range(len(A))]

def identity(n):
    """
    Tạo ma trận đơn vị (Identity Matrix) kích thước n x n.
    
    Tham số:
        n: Kích thước của ma trận vuông.
    Trả về:
        Ma trận đơn vị (đường chéo bằng 1, còn lại bằng 0).
    """
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

# ================= QR (Gram-Schmidt) =================

def qr(A):
    """
    Thực hiện phân rã QR bằng phương pháp trực giao hóa Gram-Schmidt.
    Phân tách ma trận A thành A = Q * R.
    
    Tham số:
        A: Ma trận đầu vào m x n (yêu cầu các cột độc lập tuyến tính).
    
    Trả về:
        Q: Ma trận trực chuẩn kích thước m x n (các cột vuông góc và có độ dài bằng 1).
        R: Ma trận tam giác trên kích thước n x n (đường chéo dương).
    """
    m, n = len(A), len(A[0])
    
    A_cols = [[A[i][j] for i in range(m)] for j in range(n)]
    Q_cols = []
    R = [[0.0]*n for _ in range(n)]
    
    for k in range(n):
        ak = A_cols[k][:]
        
        for j in range(k):
            R[j][k] = dot(ak, Q_cols[j])
            ak = [ak[i] - R[j][k]*Q_cols[j][i] for i in range(m)]
        
        R[k][k] = norm(ak)
        
        if R[k][k] < 1e-12:
            raise ValueError("Matrix has linearly dependent columns")
        
        # đảm bảo R_kk > 0
        if R[k][k] < 0:
            R[k][k] = -R[k][k]
            ak = [-x for x in ak]
        
        qk = [ak[i]/R[k][k] for i in range(m)]
        Q_cols.append(qk)
    
    Q = [[Q_cols[j][i] for j in range(n)] for i in range(m)]
    
    return Q, R

# ================= QR EIGEN =================

def off_diagonal_norm(A):
    """
    Tính chuẩn (tổng giá trị tuyệt đối) của tất cả các phần tử nằm ngoài đường chéo chính.
    Được dùng làm tiêu chí hội tụ trong thuật toán tìm trị riêng QR Iteration.
    
    Tham số:
        A: Ma trận vuông n x n.
    Trả về:
        Tổng giá trị tuyệt đối của các phần tử không thuộc đường chéo.
    """
    n = len(A)
    return sum(abs(A[i][j]) for i in range(n) for j in range(n) if i != j)

def qr_eigen(A, max_iter=500, tol=1e-10):
    """
    Tìm các trị riêng (eigenvalues) và xấp xỉ ma trận vector riêng (eigenvectors) 
    của ma trận A bằng thuật toán QR Iteration.
    
    Tham số:
        A: Ma trận vuông n x n.
        max_iter: Số vòng lặp tối đa.
        tol: Ngưỡng sai số (hội tụ khi chuẩn ngoài đường chéo < tol).
        
    Trả về:
        eigenvalues: Danh sách các trị riêng nằm trên đường chéo.
        eigenvectors: Ma trận vuông góc tích lũy Q_total. (Lưu ý: Q_total chỉ thực sự 
                      là ma trận vector riêng nếu A là ma trận đối xứng).
    """
    n = len(A)
    Ak = [row[:] for row in A]
    Q_total = identity(n)
    
    for _ in range(max_iter):
        Q, R = qr(Ak)
        Ak = matmul(R, Q)
        Q_total = matmul(Q_total, Q)
        
        if off_diagonal_norm(Ak) < tol:
            break
    
    eigenvalues = [Ak[i][i] for i in range(n)]
    eigenvectors = Q_total
    
    return eigenvalues, eigenvectors

# ================= NORMALIZE =================

def normalize_columns(M):
    """
    Chuẩn hóa các cột của ma trận M sao cho mỗi cột có độ dài bằng 1.
    
    Tham số:
        M: Ma trận kích thước m x n.
    Trả về:
        Ma trận có cùng kích thước với các cột đã được chuẩn hóa.
    """
    m, n = len(M), len(M[0])
    for j in range(n):
        col = [M[i][j] for i in range(m)]
        nrm = norm(col)
        if nrm < 1e-12:
            continue
        for i in range(m):
            M[i][j] /= nrm
    return M


# ================= SVD =================

def svd(A):
    """
    Thực hiện phân rã giá trị kỳ dị (Singular Value Decomposition) thu gọn.
    Phân tách ma trận A thành A = U * Σ * V^T.
    
    Tham số:
        A: Ma trận kích thước m x n.
        
    Trả về:
        U: Ma trận trực chuẩn kích thước m x n (chứa các vector kỳ dị trái).
        Sigma: Ma trận đường chéo m x n (chứa các giá trị kỳ dị giảm dần).
        Vt: Ma trận trực chuẩn kích thước n x n (chuyển vị của ma trận V).
    """
    m, n = len(A), len(A[0])
    
    At = transpose(A)
    AtA = matmul(At, A)
    
    # Eigen decomposition
    eigenvalues, V = qr_eigen(AtA)
    
    # Sort giảm dần
    idx = sorted(range(n), key=lambda i: eigenvalues[i], reverse=True)
    eigenvalues = [eigenvalues[i] for i in idx]
    V = [[V[i][j] for j in idx] for i in range(n)]
    
    # Singular values
    sigma = [math.sqrt(max(ev, 0)) for ev in eigenvalues]
    
    # Sigma matrix (m x n)
    Sigma = [[0.0]*n for _ in range(m)]
    for i in range(min(m, n)):
        Sigma[i][i] = sigma[i]
    
    # U = A V Σ^-1
    AV = matmul(A, V)
    
    U = [[0.0]*n for _ in range(m)]
    for j in range(n):
        if sigma[j] > 1e-12:
            for i in range(m):
                U[i][j] = AV[i][j] / sigma[j]
    
    # Chuẩn hóa cột U (KHÔNG dùng QR nữa)
    U = normalize_columns(U)
    
    return U, Sigma, transpose(V)

# ================= LOW-RANK APPROX =================

def low_rank_approx(U, S, Vt, k):
    """
    Tính ma trận xấp xỉ bậc thấp (Low-Rank Approximation) A_k của A,
    chỉ sử dụng k giá trị kỳ dị lớn nhất từ phân rã SVD.
    A_k = sum_{i=1}^k (σ_i * u_i * v_i^T)
    
    Tham số:
        U: Ma trận vector kỳ dị trái (m x n).
        S: Ma trận đường chéo giá trị kỳ dị (m x n).
        Vt: Ma trận vector kỳ dị phải chuyển vị (n x n).
        k: Hạng (rank) cần giữ lại (k <= n).
        
    Trả về:
        Ma trận xấp xỉ A_k kích thước m x n.
    """
    m, n = len(U), len(Vt[0])
    A_k = [[0.0]*n for _ in range(m)]
    
    for i in range(k):
        for r in range(m):
            for c in range(n):
                A_k[r][c] += S[i][i] * U[r][i] * Vt[i][c]
    
    return A_k

# ================= TEST =================

if __name__ == "__main__":
    A = [[4, 0],
         [3, -5]]
    
    print("=== QR ===")
    Q, R = qr(A)
    print("Q =", Q)
    print("R =", R)
    
    print("\n=== SVD ===")
    U, S, Vt = svd(A)
    print("U =", U)
    print("Sigma =", S)
    print("Vt =", Vt)
    
    # kiểm tra tái tạo
    A_recon = matmul(matmul(U, S), Vt)
    
    print("\nReconstructed A:")
    for row in A_recon:
        print(row)
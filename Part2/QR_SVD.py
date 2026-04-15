import math

# ================= BASIC =================

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def dot(u, v):
    return sum(x*y for x, y in zip(u, v))

def norm(v):
    return math.sqrt(dot(v, v))

def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))]
             for i in range(len(A))]

def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

# ================= QR (Gram-Schmidt) =================

def qr(A):
    """
    A = QR
    Q: m x n (trực chuẩn)
    R: n x n (tam giác trên, R_ii > 0)
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
    n = len(A)
    return sum(abs(A[i][j]) for i in range(n) for j in range(n) if i != j)

def qr_eigen(A, max_iter=500, tol=1e-10):
    """
    Tìm eigenvalues + eigenvectors bằng QR iteration
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
    A = U Σ V^T
    
    U: m x n (reduced SVD)
    Σ: m x n
    V: n x n
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
    A_k = sum_{i=1}^k σ_i u_i v_i^T
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
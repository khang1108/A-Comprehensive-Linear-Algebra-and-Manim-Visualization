import sys, os
import math

# Import hàm phụ trợ từ decomposition
from decomposition import qr_eigen

def diagonalize(A, tol=1e-7):
    """
    Chéo hóa ma trận vuông A thành A = P * D * P_inv.
    
    Nguyên lý hoạt động:
    1. Dùng thuật toán lặp QR (qr_eigen) để xấp xỉ các trị riêng (eigenvalues) của A.
    2. Gom nhóm các trị riêng giống nhau để tính Bội số đại số (Algebraic Multiplicity).
    3. Với mỗi trị riêng lambda, giải hệ phương trình thuần nhất (A - lambda*I)x = 0 
       để tìm không gian vector riêng (Eigenspace) bằng phương pháp khử Gauss.
    4. Số lượng vector riêng độc lập tìm được chính là Bội số hình học (Geometric Multiplicity).
    5. Điều kiện chéo hóa: Bội đại số phải BẰNG bội hình học cho mọi trị riêng. 
       Nếu không thỏa mãn, tung ra lỗi từ chối chéo hóa.
    6. Trả về P (chứa các vector riêng), D (đường chéo trị riêng), và P^-1.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Ma trận A phải là ma trận vuông.")

    # 1. Tìm các trị riêng bằng QR iteration
    eigenvalues, _ = qr_eigen(A)
    
    # 2. Gom nhóm các trị riêng (xử lý sai số số học bằng ngưỡng tol)
    unique_eigenvalues = []
    algebraic_mult = []
    for ev in eigenvalues:
        found = False
        for i, uev in enumerate(unique_eigenvalues):
            if abs(ev - uev) < tol:
                algebraic_mult[i] += 1
                found = True
                break
        if not found:
            unique_eigenvalues.append(ev)
            algebraic_mult.append(1)

    # Import động các hàm từ Phần 1 (Khử Gauss và Nghịch đảo) để tái sử dụng
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from part1.gaussian import gaussian_eliminate
    from part1.inverse import inverse

    eigenvectors = []
    D_diag = []

    # 3. Phân tích Eigenspace cho từng trị riêng
    for i, lam in enumerate(unique_eigenvalues):
        # Khởi tạo ma trận M = A - lambda * I
        M = [[A[r][c] - (lam if r == c else 0.0) for c in range(n)] for r in range(n)]
        
        # Đưa M về dạng bậc thang bằng khử Gauss
        U, _, _ = gaussian_eliminate(M, only_one=True)
        
        # Xác định biến cơ sở (pivots) và biến tự do (free variables)
        is_free = [True] * n
        pivots = {}
        for r in range(n):
            p_col = -1
            for c in range(n):
                if abs(U[r][c]) > tol:
                    p_col = c
                    break
            if p_col != -1:
                pivots[r] = p_col
                is_free[p_col] = False

        free_vars = [c for c in range(n) if is_free[c]]
        geometric_mult = len(free_vars)

        # 4. Kiểm tra điều kiện chéo hóa cốt lõi
        if geometric_mult < algebraic_mult[i]:
            raise ValueError(
                f"Ma trận KHÔNG chéo hóa được! Trị riêng {lam:.4g} có bội đại số là {algebraic_mult[i]} "
                f"nhưng bội hình học (số vector riêng độc lập) chỉ là {geometric_mult}."
            )

        # 5. Rút trích các vector riêng từ các biến tự do (giải hệ M*x = 0)
        for free_col in free_vars:
            vec = [0.0] * n
            vec[free_col] = 1.0  # Đặt biến tự do hiện tại = 1, các biến tự do khác = 0
            
            # Thế ngược để tìm giá trị các biến cơ sở
            for r in range(n - 1, -1, -1):
                if r not in pivots:
                    continue
                p = pivots[r]
                sum_known = sum(U[r][c] * vec[c] for c in range(p + 1, n))
                vec[p] = -sum_known / U[r][p]
            
            # Chuẩn hóa vector riêng (tùy chọn, để ma trận P đẹp hơn)
            length = math.sqrt(sum(x*x for x in vec))
            if length > 1e-12:
                vec = [x / length for x in vec]
                
            eigenvectors.append(vec)
            D_diag.append(lam)

    # 6. Lắp ráp ma trận P, D và P^-1
    # P nhận các vector riêng làm cột
    P = [[eigenvectors[c][r] for c in range(n)] for r in range(n)]
    # D là ma trận đường chéo của các trị riêng tương ứng
    D = [[D_diag[r] if r == c else 0.0 for c in range(n)] for r in range(n)]
    
    try:
        P_inv = inverse(P)
    except ValueError:
        raise ValueError("Ma trận P không khả nghịch, quá trình chéo hóa thất bại (sai số hệ thống).")

    return P, D, P_inv

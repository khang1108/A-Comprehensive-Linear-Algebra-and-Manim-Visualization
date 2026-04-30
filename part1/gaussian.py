from typing import Any, List, Optional, Tuple
import copy

# Ngưỡng sai số cho phép
_EPS = 1e-12

def gaussian_eliminate(
    A: List[List[float]],
    b: Optional[List[float]] = None,
    *,
    only_one: bool = False,
) -> Tuple[List[List[float]], List[float], int]:
    """
    Khử Gauss (Gaussian elimination) với chọn cột chủ (partial pivoting).

    Parameters:
        A: Ma trận hệ số kích thước m x n.
        b: Vector vế phải (chiều dài m). Bỏ qua nếu only_one=True.
        only_one: True thì chỉ khử A, không ghép cột b.

    Returns:
        U: Ma trận sau khử (dạng tam giác trên theo từng bước tiêu chuẩn).
        c: Vector b sau biến đổi (rỗng nếu only_one=True).
        n_swap: Số lần hoán vị hàng (dùng cho dấu định thức).
    """
    n_rows = len(A)
    n_coef = len(A[0])  # Số cột hệ số của A (không tính cột b nếu có)

    if b is None:
        b = []

    if not only_one:
        # Nếu chỉ dùng để tìm ra ma trận bậc thang của A, không cần phải mở rộng ma trận thành (A|b)
        if len(b) != n_rows:
            raise ValueError(
                "Độ dài vector b phải bằng số hàng của A (m). "
                f"Nhận được len(b)={len(b)}, n_rows={n_rows}."
            )
        mat = [list(A[i]) + [float(b[i])] for i in range(n_rows)]
        n_aug = n_coef + 1  # Chiều rộng ma trận mở rộng (A | b)
    else:
        mat = copy.deepcopy(A)
        n_aug = n_coef

    # Chỉ chọn pivot trên các cột hệ số: tránh dùng cột b làm cột pivot khi m > n
    num_pivot_cols = n_coef

    n_swap = 0
    for i in range(min(n_rows, num_pivot_cols)):
        pivot_row = i
        for j in range(i + 1, n_rows):
            if abs(mat[j][i]) > abs(mat[pivot_row][i]):
                pivot_row = j

        if i != pivot_row:
            mat[i], mat[pivot_row] = mat[pivot_row], mat[i]
            n_swap += 1

        if abs(mat[i][i]) < _EPS:
            # Cột i (từ hàng i trở xuống) gần như toàn không: bỏ qua bước khử cột này
            continue

        for j in range(i + 1, n_rows):
            mul = mat[j][i] / mat[i][i]
            for k in range(i, n_aug):
                mat[j][k] -= mul * mat[i][k]

    if only_one:
        U = mat
        c: List[float] = []
    else:
        U = [row[:n_coef] for row in mat]
        c = [row[n_aug - 1] for row in mat]

    return U, c, n_swap


def back_substitution(U: List[List[float]], c: List[float]) -> Any:
    """
    Giải hệ phương trình có ma trận dạng tam giác trên (U * x = c) bằng phương pháp thế ngược (back substitution).
    Hàm đã được nâng cấp để xử lý toàn diện mọi trường hợp: nghiệm duy nhất, vô nghiệm, và VÔ SỐ NGHIỆM.
    Trong trường hợp vô số nghiệm, hàm sẽ trả về công thức tổng quát theo các tham số tự do t_1, t_2...

    Tham số:
        U: Ma trận tam giác trên kích thước m x n.
        c: Vector vế phải độ dài m.

    Trả về:
        (nghiem, thong_bao):
            - Nếu nghiệm duy nhất: nghiem là List[float] chứa các giá trị nghiệm.
            - Nếu vô số nghiệm: nghiem là chuỗi (str) chứa công thức nghiệm tổng quát.
            - Nếu vô nghiệm: nghiem là mảng rỗng [].
            - thong_bao là chuỗi trạng thái kết quả.
    """
    n_rows = len(U)
    if n_rows == 0:
        return [], "Hệ vô nghiệm"
    n_cols = len(U[0])

    if len(c) != n_rows:
        raise ValueError(
            "Độ dài c phải bằng số hàng của U. "
            f"Nhận được len(c)={len(c)}, n_rows={n_rows}."
        )

    # 1. Quét tìm các cột Pivot (biến cơ sở) và các cột không có Pivot (biến tự do).
    # Biến tự do là những biến không có phần tử khác 0 đầu tiên trên bất kỳ hàng nào.
    pivots = {}  # Ánh xạ: chỉ số hàng -> chỉ số cột chứa pivot
    is_free = [True] * n_cols

    for i in range(n_rows):
        p_col = -1
        for j in range(n_cols):
            if abs(U[i][j]) > _EPS:
                p_col = j
                break
        
        if p_col != -1:
            # Hàng này có pivot
            pivots[i] = p_col
            is_free[p_col] = False
        else:
            # Hàng này toàn số 0. Ta kiểm tra tính nhất quán với vế phải.
            if abs(c[i]) > _EPS:
                return [], "Hệ vô nghiệm"

    # Tập hợp các chỉ số của biến tự do
    free_vars = [j for j in range(n_cols) if is_free[j]]
    num_free = len(free_vars)

    # 2. Xây dựng cấu trúc biểu thức cho từng biến x_i.
    # Nguyên lý: Mọi biến x_i sẽ được biểu diễn dưới dạng tổ hợp tuyến tính:
    # x_i = Hằng_số + Hệ_số_1 * t_1 + Hệ_số_2 * t_2 + ... + Hệ_số_k * t_k
    # Ta dùng mảng expr có kích thước (num_free + 1) để lưu hệ số này cho mỗi biến.
    # - expr[i][0] lưu giá trị Hằng_số.
    # - expr[i][k] (với k > 0) lưu hệ số của biến tự do thứ k.
    expr = [[0.0] * (num_free + 1) for _ in range(n_cols)]

    # Gán công thức mặc định cho các biến tự do (x_j = 0 + 1 * t_j)
    for idx, free_col in enumerate(free_vars):
        expr[free_col][idx + 1] = 1.0

    # 3. Quá trình thế ngược (Back Substitution).
    # Quét từ hàng dưới cùng lên trên, tính toán biểu thức cho các biến cơ sở (pivot).
    for i in range(n_rows - 1, -1, -1):
        if i not in pivots:
            continue
        
        p = pivots[i]
        
        # Giá trị bắt đầu của x_p: c[i] / U[i][p] (đóng vai trò là hằng số)
        expr[p][0] = c[i] / U[i][p]
        
        # Chuyển vế tất cả các biến đứng sau pivot: 
        # x_p = c[i]/U[i][p] - sum( (U[i][j]/U[i][p]) * x_j )
        for j in range(p + 1, n_cols):
            if abs(U[i][j]) > _EPS:
                factor = U[i][j] / U[i][p]
                
                # Trừ đi sự đóng góp của x_j vào toàn bộ biểu thức của x_p (bao gồm cả hằng số và các hệ số t_k)
                for k in range(num_free + 1):
                    expr[p][k] -= factor * expr[j][k]

    # 4. Trả về kết quả dựa trên số lượng biến tự do
    if num_free == 0:
        # Nếu không có biến tự do nào -> Nghiệm duy nhất. Lấy ra các hằng số.
        x = [expr[j][0] for j in range(n_cols)]
        return x, "Hệ có nghiệm duy nhất"
    else:
        # Nếu có biến tự do -> Vô số nghiệm. Format biểu thức nghiệm thành chuỗi dễ đọc.
        formula_lines = []
        for j in range(n_cols):
            terms = []
            
            # Ghi nhận Hằng số nếu nó khác 0, hoặc nếu toàn bộ biểu thức đều bằng 0
            if abs(expr[j][0]) > _EPS or all(abs(expr[j][k]) <= _EPS for k in range(1, num_free + 1)):
                terms.append(f"{expr[j][0]:.4g}")
            
            # Ghi nhận các hệ số của tham số tự do (t_1, t_2...)
            for k in range(num_free):
                coef = expr[j][k + 1]
                if abs(coef) > _EPS:
                    sign = "+" if coef > 0 else "-"
                    val = abs(coef)
                    
                    # Ẩn số 1 đi cho biểu thức gọn gàng (vd: 1*t_1 -> t_1)
                    if abs(val - 1.0) < _EPS:
                        term_str = f"t_{k+1}"
                    else:
                        term_str = f"{val:.4g}*t_{k+1}"
                    
                    if len(terms) > 0:
                        terms.append(f"{sign} {term_str}")
                    else:
                        if coef < 0:
                            terms.append(f"-{term_str}")
                        else:
                            terms.append(term_str)
            
            # Nếu biểu thức hoàn toàn trống, gán là 0
            if not terms:
                terms.append("0")
                
            formula_lines.append(f"x_{j+1} = " + " ".join(terms))
            
        formula_str = "\n".join(formula_lines)
        return formula_str, "Hệ có vô số nghiệm"


def verify_solution(A: List[List[float]], x: List[float], b: List[float], tol: float = 1e-9) -> bool:
    """
    Kiểm chứng kết quả nghiệm x của hệ Ax = b bằng thư viện NumPy.
    Tính toán vector phần dư r = A*x - b. Nếu chuẩn L2 của r < tol, trả về True.
    
    Tham số:
        A: Ma trận hệ số.
        x: Vector nghiệm cần kiểm chứng.
        b: Vector vế phải.
        tol: Ngưỡng sai số chấp nhận được.
        
    Trả về:
        True nếu x thực sự là nghiệm của hệ Ax = b (sai số nhỏ hơn tol), ngược lại False.
    """
    import numpy as np
    
    A_np = np.array(A, dtype=float)
    x_np = np.array(x, dtype=float)
    b_np = np.array(b, dtype=float)
    
    # Tính toán vector phần dư r = Ax - b
    residual = A_np @ x_np - b_np
    
    # Tính chuẩn L2 của phần dư
    residual_norm = float(np.linalg.norm(residual, ord=2))
    
    return residual_norm < tol

import json
import math
import os
import sys
from typing import Callable, Dict, List, Tuple, Any

import numpy as np

# Add project root to sys.path to resolve imports correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Sử dụng hàm từ file solvers của MHN
# Khang merge th này vào branch của MHN trước 
from Part3.solvers import (
    solve_gauss,
    solve_gauss_seidel,
    solve_qr_householder,
    solve_svd,
    SolverResult
)

def generate_diagonally_dominant_system(n: int) -> Tuple[List[List[float]], List[float]]:
    """
    Generate a strictly row diagonally dominant matrix A and a random vector b.
    This represents a 'well-conditioned' system.
    """
    # Use numpy for fast generation, then convert to python lists
    A_np = np.random.rand(n, n) * 10.0
    b_np = np.random.rand(n) * 10.0
    
    for i in range(n):
        row_sum = np.sum(np.abs(A_np[i, :])) - np.abs(A_np[i, i])
        A_np[i, i] = row_sum + np.random.uniform(1.0, 5.0)
        # Randomize sign for variety
        if np.random.rand() > 0.5:
            A_np[i, i] = -A_np[i, i]
            
    return A_np.tolist(), b_np.tolist()

def generate_hilbert_system(n: int) -> Tuple[List[List[float]], List[float]]:
    """
    Generate a Hilbert matrix H_n and a random vector b.
    This represents an 'ill-conditioned' system (very large condition number).
    Formula: H_{i, j} = 1 / (i + j + 1)
    """
    A = [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]
    b = (np.random.rand(n) * 10.0).tolist()
    return A, b

def vector_norm_2(v: List[float]) -> float:
    """Calculate the L2 norm (Euclidean norm) of a vector."""
    return math.sqrt(sum(x * x for x in v))

def benchmark_method(
    method_name: str,
    solver_func: Callable, 
    A: List[List[float]], 
    b: List[float], 
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Run a solver multiple times and compute average runtime and relative error.
    Leverages the SolverResult dataclass from solvers.py.
    """
    times = []
    residuals = []
    success = False
    message = ""
    
    b_norm = vector_norm_2(b)
    if b_norm < 1e-12:
        b_norm = 1.0 # Prevent division by zero
    
    # Warm-up run (to load into CPU cache)
    _ = solver_func(A, b)
    
    for _ in range(num_runs):
        # Note: solvers.py -> validate_inputs already creates a copy of A and b,
        # so we don't need to worry about the solver mutating the original data.
        result: SolverResult = solver_func(A, b)
        
        times.append(result.runtime_sec)
        residuals.append(result.residual)
        success = result.success
        message = result.message
        
    avg_time = sum(times) / num_runs
    avg_residual = sum(residuals) / num_runs
    relative_error = avg_residual / b_norm
    
    return {
        "method": method_name,
        "success": success,
        "avg_time_sec": avg_time,
        "relative_error": relative_error,
        "message": message
    }

if __name__ == "__main__":
    test_sizes = [50, 100, 200, 500] 
    # Với n=1000, code Python thuần có thể tốn khá nhiều thời gian (vài phút)
    # Thêm 1000 vào list sau khi test ok các size nhỏ
    
    solvers = {
        "Gauss Elimination": solve_gauss,
        "Gauss-Seidel": solve_gauss_seidel,
        "QR-Householder": solve_qr_householder,
        "SVD": solve_svd
    }
    
    matrix_generators = {
        "Well-Conditioned (Diagonally Dominant)": generate_diagonally_dominant_system,
        "Ill-Conditioned (Hilbert)": generate_hilbert_system
    }
    
    all_benchmark_data = []

    print(f"{'Size':<5} | {'Matrix Type':<40} | {'Method':<20} | {'Avg Time (s)':<15} | {'Relative Error'}")
    print("-" * 110)

    for n in test_sizes:
        for type_name, gen_func in matrix_generators.items():
            A, b = gen_func(n)
            
            size_result = {
                "n": n,
                "matrix_type": type_name,
                "results": []
            }
            
            for method_name, solver_func in solvers.items():
                try:
                    res = benchmark_method(method_name, solver_func, A, b, num_runs=5)
                    size_result["results"].append(res)
                    
                    status_flag = "Pass" if res['success'] else "Fail"
                    print(f"{n:<5} | {type_name:<40} | {method_name:<20} | {res['avg_time_sec']:<15.6e} | {res['relative_error']:<15.6e} ({status_flag})")
                except Exception as e:
                    print(f"{n:<5} | {type_name:<40} | {method_name:<20} | {'FAILED':<15} | {str(e)}")
            
            all_benchmark_data.append(size_result)
            print("-" * 110)
            
    # Export data to JSON for Jupyter Notebook analysis (for Nghia)
    output_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_data, f, indent=4)
        
    print(f"\n[INFO] Benchmark complete. Data exported to: {output_file}")
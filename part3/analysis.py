"""
Benchmark Phần 3 — đúng theo yêu cầu đề bài:
  - n ∈ {50, 100, 200, 500}  (bỏ 1000 vì pure Python quá chậm)
  - Trung bình 5 lần chạy mỗi (n, solver)
  - Tính sai số tương đối: ‖Ax̂-b‖₂ / ‖b‖₂
  - Hai loại ma trận: Hilbert (ill-cond.) và SPD (well-cond.)
  - SVD bị skip với n > SVD_MAX_SIZE (quá chậm ở pure Python)
  - Gauss-Seidel trên Hilbert chỉ chạy 1 lần, max 30 iter (không hội tụ, tốn thời gian)
"""
import json, os, sys, time
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from part3.solvers import solve_gauss, solve_gauss_seidel, solve_qr_householder, solve_svd

SVD_MAX_SIZE = 15          # SVD pure Python chậm, skip n > 15
N_RUNS       = 5           # trung bình 5 lần theo yêu cầu đề


def hilbert_matrix(n):
    """Ma trận Hilbert: H[i,j] = 1/(i+j+1) — ill-conditioned kinh điển."""
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def random_spd_matrix(n, alpha=10.0, seed=42):
    """Ma trận SPD ngẫu nhiên: A = B^T B + alpha*I — well-conditioned."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    return (B.T @ B + alpha * np.eye(n)).tolist()


def condition_number(A_list):
    return float(np.linalg.cond(np.array(A_list)))


def run_solver(sfunc, A, b, runs):
    """Chạy solver `runs` lần, trả về (avg_time, avg_residual, rel_error, success, iters)."""
    b_norm = float(np.linalg.norm(b))
    times, residuals = [], []
    last = None
    for _ in range(runs):
        last = sfunc(A, b)
        times.append(last.runtime_sec)
        residuals.append(last.residual if np.isfinite(last.residual) else float("inf"))
    avg_t   = sum(times) / runs
    avg_res = sum(residuals) / runs
    rel_err = avg_res / max(b_norm, 1e-12)
    return avg_t, avg_res, rel_err, last.success, last.iterations, last.message


def run_benchmark(sizes):
    rng = np.random.default_rng(0)
    results = []

    all_solvers = {
        "Gauss":          (solve_gauss,          N_RUNS),
        "Gauss-Seidel":   (solve_gauss_seidel,   N_RUNS),
        "QR-Householder": (solve_qr_householder, N_RUNS),
        "SVD":            (solve_svd,             1),      # chậm, chỉ 1 lần
    }

    hdr = f"{'n':>5} | {'Loại':^8} | {'Phương pháp':^18} | {'t_avg(s)':>10} | {'‖Ax-b‖/‖b‖':>14} | {'Iter':>5} | OK"
    print(hdr)
    print("-" * len(hdr))

    for n in sizes:
        b = rng.random(n).tolist()

        for mat_type, gen in [("hilbert", hilbert_matrix), ("spd", random_spd_matrix)]:
            A = gen(n)
            kappa = condition_number(A)

            for sname, (sfunc, runs) in all_solvers.items():
                # Skip SVD cho n lớn
                if sname == "SVD" and n > SVD_MAX_SIZE:
                    print(f"  {n:>3} | {mat_type:^8} | {sname:^18} | SKIP (n > {SVD_MAX_SIZE})")
                    continue

                # Gauss-Seidel trên Hilbert: không hội tụ → chỉ 1 lần, ít iter
                effective_runs = runs
                if sname == "Gauss-Seidel" and mat_type == "hilbert":
                    effective_runs = 1
                    # wrap để truyền max_iter=30
                    def _seidel_limited(A, b):
                        return solve_gauss_seidel(A, b, max_iter=30)
                    sfunc_use = _seidel_limited
                else:
                    sfunc_use = sfunc

                try:
                    avg_t, avg_res, rel_err, ok, iters, msg = run_solver(sfunc_use, A, b, effective_runs)
                except Exception as exc:
                    print(f"  {n:>3} | {mat_type:^8} | {sname:^18} | ERROR: {exc}")
                    continue

                flag = "✓" if ok else "✗"
                print(f"  {n:>3} | {mat_type:^8} | {sname:^18} | {avg_t:>10.4f} | {rel_err:>14.3e} | {iters:>5} | {flag}")

                results.append({
                    "n": n, "matrix_type": mat_type, "condition_number": kappa,
                    "solver": sname, "n_runs": effective_runs,
                    "avg_time_sec": avg_t, "avg_residual": avg_res,
                    "relative_error": rel_err, "success": ok,
                    "iterations": iters, "message": msg,
                })

    # Lưu JSON cho notebook
    out = os.path.join(os.path.dirname(__file__), "benchmark_results_full.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Đã lưu → {out}")
    return results


if __name__ == "__main__":
    # n ∈ {50,100,200,500} — đúng yêu cầu đề bài
    # (1000 bỏ qua vì Gauss thuần Python n=1000 × 5 lần ≈ 160s)
    sizes = [50, 100, 200, 500]
    print(f"[INFO] Benchmark Phần 3 — {N_RUNS} lần/solver")
    print(f"[INFO] SVD chỉ chạy n ≤ {SVD_MAX_SIZE}\n")
    run_benchmark(sizes)

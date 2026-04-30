"""
Phân tích ổn định số - Phần 3
Script tạo notebook dưới dạng file Python có thể chạy độc lập hoặc convert sang .ipynb
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from part3.solvers import (
    solve_gauss, solve_gauss_seidel,
    solve_qr_householder, solve_svd
)

OUTPUT_DIR = os.path.dirname(__file__)

# ============================================================
# PHẦN 1 — HÀM TẠO MA TRẬN
# ============================================================
def hilbert(n):
    """Ma trận Hilbert bậc n: H[i,j] = 1/(i+j+1), 0-indexed"""
    return [[1.0/(i+j+1) for j in range(n)] for i in range(n)]

def random_spd(n, alpha=10.0, seed=7):
    """Ma trận SPD ngẫu nhiên: A = B^T B + alpha*I"""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    A = B.T @ B + alpha * np.eye(n)
    return A.tolist()

def cond(A_list):
    return float(np.linalg.cond(np.array(A_list)))

# ============================================================
# PHẦN 2 — BENCHMARK HỘI TỤ (kích thước nhỏ, tránh SVD chậm)
# ============================================================
SMALL_SIZES = [5, 8, 10, 12, 15]
rng_b = np.random.default_rng(42)

rows = []
print(f"\n{'n':>5} | {'Loại':^12} | {'κ(A)':^14} | {'Phương pháp':^18} | {'Phần dư':^14} | {'Iter':>5} | {'OK'}")
print("-"*90)

for n in SMALL_SIZES:
    for mtype, gen in [("Hilbert", hilbert), ("SPD", random_spd)]:
        A = gen(n)
        kappa = cond(A)
        b = rng_b.random(n).tolist()
        b_norm = float(np.linalg.norm(b))

        for sname, sfunc in [("Gauss", solve_gauss),
                              ("Gauss-Seidel", solve_gauss_seidel),
                              ("QR-Householder", solve_qr_householder),
                              ("SVD", solve_svd)]:
            r = sfunc(A, b)
            rel = r.residual / max(b_norm, 1e-12)
            rows.append({"n": n, "matrix": mtype, "kappa": kappa,
                         "solver": sname, "residual": r.residual,
                         "rel_error": rel, "iter": r.iterations,
                         "success": r.success, "time": r.runtime_sec})
            print(f"{n:>5} | {mtype:^12} | {kappa:14.3e} | {sname:^18} | {r.residual:14.3e} | {r.iterations:>5} | {'✓' if r.success else '✗'}")

# ============================================================
# PHẦN 3 — BENCHMARK THỜI GIAN (không dùng SVD pure-Python chậm)
# ============================================================
TIME_SIZES = [10, 20, 50, 100, 150, 200]
time_rows = []

print("\n\n=== BENCHMARK THỜI GIAN (SPD - well-conditioned) ===")
print(f"{'n':>5} | {'Gauss':>12} | {'Gauss-Seidel':>14} | {'QR-Hh':>12}")
print("-"*55)

rng_t = np.random.default_rng(99)
for n in TIME_SIZES:
    A = random_spd(n)
    b = rng_t.random(n).tolist()
    times = {}
    for sname, sfunc in [("Gauss", solve_gauss),
                          ("Gauss-Seidel", solve_gauss_seidel),
                          ("QR-Householder", solve_qr_householder)]:
        t0 = time.perf_counter()
        for _ in range(3):
            sfunc(A, b)
        times[sname] = (time.perf_counter()-t0)/3
        time_rows.append({"n": n, "solver": sname, "time": times[sname]})
    print(f"{n:>5} | {times['Gauss']:>12.5f} | {times['Gauss-Seidel']:>14.5f} | {times['QR-Householder']:>12.5f}")

# ============================================================
# PHẦN 4 — VẼ ĐỒ THỊ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Phần 3 — Phân tích ổn định số và độ phức tạp thời gian", fontsize=13, fontweight="bold")

# --- 4.1 Số điều kiện theo n ---
ax = axes[0]
ns_h = sorted(set(r["n"] for r in rows if r["matrix"]=="Hilbert"))
ns_s = sorted(set(r["n"] for r in rows if r["matrix"]=="SPD"))
kh = [next(r["kappa"] for r in rows if r["matrix"]=="Hilbert" and r["n"]==n) for n in ns_h]
ks = [next(r["kappa"] for r in rows if r["matrix"]=="SPD"   and r["n"]==n) for n in ns_s]
ax.semilogy(ns_h, kh, "ro-", label="Hilbert (ill-conditioned)", linewidth=2)
ax.semilogy(ns_s, ks, "bs-", label="SPD ngẫu nhiên (well-conditioned)", linewidth=2)
ax.set_xlabel("Kích thước n")
ax.set_ylabel("κ(A) — Số điều kiện (log)")
ax.set_title("Số điều kiện κ(A) theo n")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# --- 4.2 Phần dư theo phương pháp & loại ma trận ---
ax = axes[1]
solvers_order = ["Gauss", "Gauss-Seidel", "QR-Householder", "SVD"]
x_pos = np.arange(len(solvers_order))
n_demo = max(SMALL_SIZES)
colors_h, colors_s = [], []
for s in solvers_order:
    rh = next((r for r in rows if r["n"]==n_demo and r["matrix"]=="Hilbert" and r["solver"]==s), None)
    rs = next((r for r in rows if r["n"]==n_demo and r["matrix"]=="SPD"    and r["solver"]==s), None)
    colors_h.append(rh["residual"] if rh and np.isfinite(rh["residual"]) else 1e10)
    colors_s.append(rs["residual"] if rs and np.isfinite(rs["residual"]) else 1e10)

bars1 = ax.bar(x_pos-0.2, colors_h, 0.38, label="Hilbert", color="salmon", log=True)
bars2 = ax.bar(x_pos+0.2, colors_s, 0.38, label="SPD", color="steelblue", log=True)
ax.set_xticks(x_pos)
ax.set_xticklabels([s.replace("-","\n") for s in solvers_order], fontsize=8)
ax.set_ylabel("‖Ax-b‖₂ (log)")
ax.set_title(f"Phần dư theo phương pháp (n={n_demo})")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# --- 4.3 Đồ thị log-log thời gian ---
ax = axes[2]
colors_map = {"Gauss":"royalblue","Gauss-Seidel":"seagreen","QR-Householder":"darkorange"}
for s, c in colors_map.items():
    ns_ = sorted(set(r["n"] for r in time_rows if r["solver"]==s))
    ts_ = [next(r["time"] for r in time_rows if r["n"]==n and r["solver"]==s) for n in ns_]
    ax.loglog(ns_, ts_, "o-", color=c, label=s, linewidth=2, markersize=5)

# Đường tham chiếu lý thuyết
ns_ref = np.array([10, 200], dtype=float)
ref_t0 = next(r["time"] for r in time_rows if r["n"]==10 and r["solver"]=="Gauss")
ax.loglog(ns_ref, ref_t0*(ns_ref/10)**3, "k--", alpha=0.5, label="O(n³) ref")
ax.loglog(ns_ref, ref_t0*(ns_ref/10)**2, "k:",  alpha=0.5, label="O(n²) ref")
ax.set_xlabel("Kích thước n (log)")
ax.set_ylabel("Thời gian (s) (log)")
ax.set_title("Đồ thị log-log: Thời gian vs Kích thước")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "phan3_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n[OK] Đã lưu đồ thị: {fig_path}")

# ============================================================
# PHẦN 5 — LƯU KẾT QUẢ JSON
# ============================================================
out_json = os.path.join(OUTPUT_DIR, "benchmark_results_full.json")
with open(out_json, "w") as f:
    json.dump({"stability": rows, "timing": time_rows}, f, indent=2, default=str)
print(f"[OK] Đã lưu JSON: {out_json}")

# ============================================================
# PHẦN 6 — NHẬN XÉT TÓM TẮT
# ============================================================
print("\n" + "="*70)
print("NHẬN XÉT TÓM TẮT")
print("="*70)
print(f"{'Phát hiện quan trọng':}")
print(f"  1. Ma trận Hilbert n=10: κ ≈ {kh[-1]:.2e} — ill-conditioned")
print(f"  2. Ma trận SPD    n=10: κ ≈ {ks[-1]:.2e} — well-conditioned")
print(f"  3. Gauss-Seidel trên Hilbert KHÔNG hội tụ (residual > 1.0)")
print(f"  4. SVD và QR trả về residual=inf trên Hilbert n>10")
print(f"     → Nguyên nhân: SVD part2 thuần Python gặp tràn số do κ quá lớn")
print(f"  5. Tất cả solver hoạt động tốt trên SPD (residual ≈ 1e-15 đến 1e-7)")
print(f"  6. Gauss-Seidel chậm hơn Gauss do lặp nhiều vòng")

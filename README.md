# Đồ Án 1: Ma Trận và Nền Tảng Khoa Học Máy Tính
*(Project 1: Matrices and Foundations of Computational Science)*

Đây là mã nguồn và báo cáo chính thức cho Đồ án 1 của môn học. Đồ án tập trung vào việc tự cài đặt (from scratch) các phương pháp nền tảng của Đại số tuyến tính tính toán (Phép khử Gauss, Phân rã QR, Phân rã SVD) và đánh giá hiệu năng/độ ổn định của chúng trên thực tế.

Link Repo: [https://github.com/khang1108/A-Comprehensive-Linear-Algebra-and-Manim-Visualization](https://github.com/khang1108/A-Comprehensive-Linear-Algebra-and-Manim-Visualization)

## Thông Tin Nhóm
- **Khang Phuc Nguyen** (24120068) - *Leader*
- **Nghia Trong Hoang** (24120103)
- **Nhat Hoang Mai** (24120109)
- **Nhat Hoang Nguyen** (24120110)
- **Long Nhat Phung Vo** (24120088)

---

## Cấu Trúc Thư Mục
Dự án được cấu trúc thành 3 phần chính và 1 thư mục báo cáo đúng theo yêu cầu của đồ án:

```text
├── README.md                 # File hướng dẫn (bạn đang đọc file này)
├── requirements.txt          # Các thư viện Python cần thiết (numpy, matplotlib,...)
├── requirements-manim.txt    # Các thư viện phụ trợ để chạy Manim (tùy chọn)
├── part1/                    # PHẦN 1: Phép khử Gauss và các ứng dụng
│   ├── gaussian.py           # Khử Gauss, tìm nghiệm (xử lý Vô số nghiệm)
│   ├── determinant.py        # Tính định thức
│   ├── inverse.py            # Tìm ma trận nghịch đảo (Gauss-Jordan)
│   ├── rank_basis.py         # Tìm Hạng và Cơ sở không gian (row, col, null)
│   └── part1_demo.ipynb      # Notebook trình diễn & kiểm chứng kết quả với NumPy
├── part2/                    # PHẦN 2: Phân rã ma trận và Trực quan hóa
│   ├── decomposition.py      # Phân rã QR, SVD từ đầu (không dùng thư viện)
│   ├── diagonalization.py    # Chéo hóa ma trận (P D P^-1)
│   ├── manim_scene.py        # Script kiểm chứng các phép phân rã
│   ├── intro_scene.py        # Script Manim: Intro giới thiệu nhóm
│   ├── build_video.sh        # Bash script: Render Intro & ghép vào Demo
│   └── demo_video.mp4        # Video Manim trình bày trực quan thuật toán
├── part3/                    # PHẦN 3: Phân tích hiệu năng & Tính ổn định
│   ├── solvers.py            # Tổng hợp các hàm giải hệ phương trình
│   ├── benchmark.py          # Script chạy test, đo đạc thời gian & sai số
│   ├── benchmark_results_full.json # Dữ liệu raw sinh ra từ benchmark
│   └── analysis.ipynb        # Phân tích dữ liệu & vẽ biểu đồ (log-log, condition number)
└── report/                   # BÁO CÁO (LaTeX)
    ├── report.pdf            # File báo cáo hoàn chỉnh (định dạng PDF)
    ├── report.tex            # Mã nguồn LaTeX chính
    └── sections/             # Các chương/mục nhỏ được chia tách ra
```

---

## Hướng Dẫn Cài Đặt

Môi trường code được kiểm thử trên **Python 3.10+**. Tuy nhóm không sử dụng NumPy/SciPy/SymPy cho các thuật toán cốt lõi, nhưng chúng vẫn được dùng để kiểm chứng (verify) và phân tích dữ liệu (plot) trong notebook.

1. **Tạo môi trường ảo (khuyến nghị):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
   ```

2. **Cài đặt các thư viện cơ bản:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Tùy chọn) Cài đặt để render video Manim (Phần 2):**
   ```bash
   pip install -r requirements-manim.txt
   ```
   *Lưu ý: Manim cần FFmpeg và LaTeX (hoặc MiKTeX) được cài đặt sẵn trên hệ điều hành.*

---

## Hướng Dẫn Chạy Từng Phần

### Phần 1: Khử Gauss & Ứng dụng
Toàn bộ thuật toán được cài đặt độc lập. Để xem minh họa chi tiết từng test case edge (chẳng hạn hệ phương trình có vô số nghiệm) cùng với việc đối chiếu bằng NumPy:
- Mở và chạy toàn bộ cell trong file `part1/part1_demo.ipynb`.

### Phần 2: Phân rã ma trận (QR/SVD) & Video Manim
Các hàm cốt lõi nằm ở `decomposition.py` và `diagonalization.py`.
- **Xem Video Trực Quan trên YouTube:** [https://youtu.be/-fa9aWKXB0Y](https://youtu.be/-fa9aWKXB0Y)
- Hoặc bạn có thể xem file local: `part2/demo_video.mp4`.

### Phần 3: Phân tích tính ổn định
Phần này kiểm tra xem thuật toán của chúng ta hoạt động thế nào trên **Ma trận SPD** (tốt) và **Ma trận Hilbert** (ill-conditioned).
- Chạy thực nghiệm (Sẽ mất khoảng vài phút tùy CPU):
  ```bash
  cd part3
  python benchmark.py
  ```
- Xem biểu đồ Log-log và kết quả sai số tích lũy:
  Mở và chạy file `part3/analysis.ipynb`.

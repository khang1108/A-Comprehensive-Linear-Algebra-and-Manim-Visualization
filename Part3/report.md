# Báo cáo Phần 3 - Nhận xét, Kết luận, Trình bày

## 1. Mục tiêu của Part 3

Part 3 không chỉ dừng ở việc "code chạy được", mà tập trung vào phần đánh giá thực nghiệm và rút ra kết luận học thuật. Cụ thể, các mục tiêu gồm:

1. Kiểm chứng độ đúng của các hàm solver đã triển khai trong `Part3/solvers.py`.
2. So sánh kết quả của các solver tự cài đặt với thư viện chuẩn (NumPy).
3. Đánh giá chất lượng nghiệm thông qua các chỉ số sai số và residual.
4. Phân tích hiệu năng tương đối của từng phương pháp (thời gian chạy, số vòng lặp).
5. Đánh giá tính ổn định số và khả năng xử lý các trường hợp khó (ma trận gần suy biến, suy biến).
6. Rút ra khuyến nghị: trong bài toán nào nên dùng Gauss, Gauss-Seidel, QR-Householder, hoặc SVD.

---

## 2. Các phương pháp được đánh giá

Trong file `Part3/solvers.py`, hệ phương trình tuyến tính

\[
Ax = b
\]

được giải bằng 4 phương pháp:

### 2.1. Gaussian Elimination (Gauss)

- Ý tưởng: Khử tiến để đưa hệ về dạng tam giác trên, sau đó thế lùi để tìm nghiệm.
- Điểm mạnh:
  - Dễ hiểu, phổ biến, phù hợp hệ vuông không suy biến.
  - Cho nghiệm chính xác cao trong nhiều bài toán cơ bản.
- Hạn chế:
  - Có thể nhạy cảm số học nếu pivot nhỏ hoặc ma trận điều kiện xấu.
  - Không tối ưu cho các trường hợp suy biến.

### 2.2. Gauss-Seidel (phương pháp lặp)

- Công thức lặp:

\[
 x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j<i} a_{ij}x_j^{(k+1)} - \sum_{j>i} a_{ij}x_j^{(k)}\right)
\]

- Điểm mạnh:
  - Đơn giản, hiệu quả với một số lớp ma trận phù hợp.
  - Hữu ích khi muốn dừng sớm theo ngưỡng sai số.
- Hạn chế:
  - Không phải lúc nào cũng hội tụ.
  - Hội tụ phụ thuộc cấu trúc ma trận (ví dụ: chéo trội nghiêm ngặt).

### 2.3. QR decomposition (Householder)

- Ý tưởng:
  1. Phân rã \(A = QR\) bằng phản xạ Householder.
  2. Giải \(Rx = Q^Tb\).
- Điểm mạnh:
  - Ổn định số tốt hơn nhiều kỹ thuật khử trực tiếp đơn giản.
  - Phù hợp bài toán cần độ bền số cao.
- Hạn chế:
  - Cài đặt phức tạp hơn Gauss.
  - Chi phí tính toán vẫn lớn với ma trận lớn.

### 2.4. SVD (Moore-Penrose pseudo-inverse)

- Ý tưởng:
  - Phân rã \(A = U\Sigma V^T\), rồi dùng giả nghịch đảo \(A^+ = V\Sigma^+U^T\) để suy ra nghiệm.
- Điểm mạnh:
  - Rất ổn định số.
  - Xử lý tốt ma trận suy biến hoặc gần suy biến.
- Hạn chế:
  - Chi phí tính toán thường cao nhất trong 4 phương pháp.

---

## 3. Tiêu chí đánh giá

Để đánh giá khách quan, báo cáo dùng các chỉ số sau:

1. Sai khác nghiệm so với chuẩn thư viện:

\[
\|x - x_{ref}\|_2
\]

Trong đó \(x_{ref}\) là nghiệm tham chiếu từ NumPy (`np.linalg.solve` hoặc `np.linalg.lstsq`).

2. Sai số residual:

\[
\|Ax - b\|_2
\]

Residual nhỏ cho thấy nghiệm thỏa hệ tốt.

3. Thời gian chạy (`runtime_sec`):
- So sánh tương đối giữa các phương pháp.
- Không kết luận tuyệt đối theo mili-giây nếu chưa benchmark nhiều lần.

4. Số vòng lặp (`iterations`):
- Quan trọng với Gauss-Seidel.
- Kết hợp với trạng thái hội tụ để đánh giá tính khả thi.

5. Trạng thái thành công (`success`) và thông báo (`message`):
- Dùng để hiểu rõ phương pháp thất bại do nguyên nhân nào.

---

## 4. Thiết kế thực nghiệm

### 4.1. Bộ dữ liệu kiểm thử

Nên có ít nhất 4 nhóm test:

1. **Hệ nhỏ, điều kiện tốt**:
- Mục tiêu: kiểm tra đúng/sai cơ bản.
- Kỳ vọng: cả 4 phương pháp cho kết quả gần nhau.

2. **Hệ chéo trội nghiêm ngặt**:
- Mục tiêu: tạo môi trường thuận lợi cho Gauss-Seidel hội tụ.
- Kỳ vọng: Gauss-Seidel hội tụ ổn định và sai số nhỏ.

3. **Hệ gần suy biến**:
- Mục tiêu: kiểm tra độ ổn định số.
- Kỳ vọng: SVD và QR thường ổn định hơn.

4. **Hệ suy biến**:
- Mục tiêu: quan sát hành vi khi ma trận không khả nghịch.
- Kỳ vọng: một số phương pháp direct thất bại là bình thường; SVD thường vẫn đưa ra nghiệm least-squares hợp lý.

### 4.2. Cách chạy

- Chạy notebook `Part3/check.ipynb` từ đầu đến cuối.
- Lưu toàn bộ output bảng kết quả.
- Trích xuất các dòng quan trọng để đưa vào báo cáo.

---

## 5. Quy trình phân tích kết quả

Sau khi chạy notebook, phân tích theo thứ tự sau:

1. So sánh `x_diff_vs_numpy`:
- Nếu rất nhỏ (ví dụ \(10^{-8}\) đến \(10^{-12}\)), nghiệm gần như trùng tham chiếu.
- Nếu lớn, cần kiểm tra điều kiện ma trận hoặc logic cài đặt.

2. So sánh `residual_reported` và `residual_recomputed`:
- Hai giá trị nên gần nhau.
- Nếu chênh lệch lớn, có thể có lỗi tính residual hoặc lỗi kiểu dữ liệu.

3. Đọc `success` và `message`:
- Không kết luận chỉ dựa vào 1 cột residual.
- Trạng thái hội tụ/không hội tụ cần được giải thích theo lý thuyết phương pháp.

4. Xem `iterations` của Gauss-Seidel:
- Nếu quá lớn hoặc chạm `max_iter`, khả năng hội tụ kém.
- Nếu hội tụ nhanh trên case chéo trội, điều này phù hợp lý thuyết.

5. Xem `runtime_sec`:
- Dùng để so thứ hạng tương đối trong từng case.
- Không dùng một lần đo duy nhất để kết luận cuối cùng về hiệu năng tổng quát.

---

## 6. Nhận xét mẫu chi tiết (khung viết sẵn)

Phần này bạn có thể điền số liệu thật lấy từ notebook.

### 6.1. Case 1 - small_well_conditioned

- Kết quả quan sát:
  - Gauss: (điền số liệu)
  - Gauss-Seidel: (điền số liệu)
  - QR-Householder: (điền số liệu)
  - SVD: (điền số liệu)
- Nhận xét:
  - Các phương pháp direct thường cho nghiệm gần trùng chuẩn NumPy.
  - Residual đều rất nhỏ, chứng tỏ hệ được giải đúng.

### 6.2. Case 2 - strictly_diagonally_dominant

- Kết quả quan sát:
  - Gauss-Seidel hội tụ sau (điền iterations) vòng.
  - Sai khác nghiệm với NumPy là (điền giá trị).
- Nhận xét:
  - Trường hợp này xác nhận điều kiện chéo trội giúp phương pháp lặp hoạt động tốt.
  - Các phương pháp direct vẫn ổn định, nhưng không có lợi thế rõ rệt về tính lặp.

### 6.3. Case 3 - near_singular

- Kết quả quan sát:
  - Sai số một số phương pháp tăng hơn so với case điều kiện tốt.
  - SVD thường giữ tính ổn định tốt hơn.
- Nhận xét:
  - Đây là case quan trọng để đánh giá độ bền số.
  - Nên nhấn mạnh sự khác biệt giữa "chạy được" và "ổn định số".

### 6.4. Case 4 - singular

- Kết quả quan sát:
  - Một số solver có thể trả `success=False`.
  - SVD vẫn có thể trả nghiệm theo nghĩa least-squares.
- Nhận xét:
  - Thất bại của một số phương pháp ở case này là đúng kỳ vọng lý thuyết, không phải lỗi code.
  - Cần diễn giải rõ để tránh hiểu nhầm khi chấm báo cáo.

---

## 7. Kết luận tổng hợp

### 7.1. Kết luận về độ đúng

- Trên các hệ không suy biến, các phương pháp direct (Gauss, QR, SVD) thường cho nghiệm rất gần chuẩn thư viện.
- Gauss-Seidel có thể cho nghiệm tốt nếu hội tụ, đặc biệt trên ma trận chéo trội.

### 7.2. Kết luận về ổn định số

- QR-Householder và SVD có xu hướng ổn định hơn trên các case khó.
- SVD là lựa chọn mạnh cho ma trận suy biến hoặc gần suy biến.

### 7.3. Kết luận về hiệu năng

- Gauss thường có tốc độ tốt với hệ vừa và nhỏ.
- Gauss-Seidel phụ thuộc số vòng lặp; có thể nhanh hoặc chậm tùy case.
- SVD thường tốn thời gian hơn nhưng đổi lại tính linh hoạt và ổn định.

### 7.4. Khuyến nghị sử dụng

1. Hệ vuông khả nghịch, bài toán tiêu chuẩn: ưu tiên Gauss hoặc QR.
2. Ưu tiên ổn định số: chọn QR-Householder.
3. Hệ suy biến/gần suy biến: chọn SVD.
4. Muốn dùng phương pháp lặp: chỉ dùng Gauss-Seidel khi có điều kiện hội tụ phù hợp.

---

## 8. Hạn chế của bài làm hiện tại

1. Kết quả thời gian chạy có thể phụ thuộc máy và điều kiện môi trường.
2. Số lượng test case ngẫu nhiên chưa đủ lớn để kết luận thống kê mạnh.
3. Chưa benchmark lặp nhiều lần để lấy trung bình và độ lệch chuẩn.
4. Chưa mở rộng sang hệ rất lớn hoặc sparse matrix.

---

## 9. Hướng phát triển tiếp theo

1. Thêm benchmark lặp nhiều lần và lấy trung bình runtime.
2. Thêm biểu đồ trực quan (boxplot runtime, log-scale residual).
3. Thêm các phương pháp iterative khác (Jacobi, Conjugate Gradient cho SPD).
4. Thêm preconditioning để cải thiện hội tụ.
5. Viết unit test tự động cho từng solver và từng loại ma trận.

---

## 10. Phụ lục: Mẫu bảng đưa vào báo cáo

Bạn có thể copy bảng sau và điền số liệu thực tế.

### 10.1. Bảng độ chính xác

| Case | Method | Success | $\|x-x_{ref}\|_2$ | $\|Ax-b\|_2$ |
|---|---|---:|---:|---:|
| small_well_conditioned | gauss |  |  |  |
| small_well_conditioned | gauss_seidel |  |  |  |
| small_well_conditioned | qr_householder |  |  |  |
| small_well_conditioned | svd |  |  |  |
| strictly_diagonally_dominant | gauss |  |  |  |
| strictly_diagonally_dominant | gauss_seidel |  |  |  |
| strictly_diagonally_dominant | qr_householder |  |  |  |
| strictly_diagonally_dominant | svd |  |  |  |
| near_singular | gauss |  |  |  |
| near_singular | gauss_seidel |  |  |  |
| near_singular | qr_householder |  |  |  |
| near_singular | svd |  |  |  |
| singular | gauss |  |  |  |
| singular | gauss_seidel |  |  |  |
| singular | qr_householder |  |  |  |
| singular | svd |  |  |  |

### 10.2. Bảng hiệu năng

| Case | Method | Runtime (s) | Iterations |
|---|---|---:|---:|
| small_well_conditioned | gauss |  |  |
| small_well_conditioned | gauss_seidel |  |  |
| small_well_conditioned | qr_householder |  |  |
| small_well_conditioned | svd |  |  |
| strictly_diagonally_dominant | gauss |  |  |
| strictly_diagonally_dominant | gauss_seidel |  |  |
| strictly_diagonally_dominant | qr_householder |  |  |
| strictly_diagonally_dominant | svd |  |  |
| near_singular | gauss |  |  |
| near_singular | gauss_seidel |  |  |
| near_singular | qr_householder |  |  |
| near_singular | svd |  |  |
| singular | gauss |  |  |
| singular | gauss_seidel |  |  |
| singular | qr_householder |  |  |
| singular | svd |  |  |

---

## 11. Checklist trước khi nộp

1. Notebook `Part3/check.ipynb` chạy từ trên xuống không lỗi.
2. Có output bảng kết quả rõ ràng cho tất cả test case.
3. Báo cáo markdown có đủ: mục tiêu, phương pháp, thực nghiệm, nhận xét, kết luận.
4. Mọi kết luận đều có số liệu đi kèm.
5. Trình bày sạch, chia section rõ ràng, công thức đúng.
6. Nêu rõ hạn chế và hướng phát triển để thể hiện tư duy phản biện.

---

## 12. References

1. Gilbert Strang, *Introduction to Linear Algebra*.
2. Trefethen and Bau, *Numerical Linear Algebra*.
3. NumPy documentation:
   - `numpy.linalg.solve`
   - `numpy.linalg.lstsq`
   - `numpy.linalg.svd`
4. Giáo trình môn Đại số tuyến tính và Phương pháp số của môn học.

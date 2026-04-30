# KỊCH BẢN VIDEO MANIM — PHẦN 2: PHÂN RÃ QR VÀ SVD

> **Thời lượng mục tiêu:** 15–25 phút  
> **Đối tượng:** Sinh viên đại học năm 2+, có kiến thức cơ bản về ma trận, vector, và phép khử Gauss.  
> **Mục tiêu:** Truyền tải trực quan & có chiều sâu về hai phương pháp phân rã ma trận quan trọng nhất: **QR** (Gram–Schmidt) và **SVD** (Singular Value Decomposition).

---

## HƯỚNG DẪN CHUNG CHO AI SINH CODE MANIM

### Theme & Phong cách

| Thuộc tính | Giá trị |
|---|---|
| **Nền (background)** | Xám đậm `#1e1e1e` |
| **Chữ chính** | Trắng `#FFFFFF`, font size `36` (body), `44` (heading) |
| **Chữ phụ / chú thích** | Xám nhạt `#AAAAAA`, font size `28` |
| **Công thức LaTeX** | Trắng `#FFFFFF`, font size mặc định MathTex |
| **Màu nhấn 1 (primary accent)** | Xanh dương `#58A6FF` — dùng cho vector, cạnh, highlight |
| **Màu nhấn 2 (secondary accent)** | Cam `#FFA657` — dùng cho vector thứ hai, kết quả |
| **Màu nhấn 3** | Xanh lá `#7EE787` — dùng cho kết quả đúng, check mark |
| **Màu nhấn 4** | Tím `#D2A8FF` — dùng cho eigenvalue, singular value |
| **Màu cảnh báo** | Đỏ nhạt `#FF7B72` — dùng cho lỗi, chú ý quan trọng |
| **Đường kẻ trục / lưới** | Xám trung `#444444` |
| **Tốc độ animation mặc định** | `run_time=1.5` cho Transform, `run_time=1.0` cho FadeIn/FadeOut |

### Quy tắc bố cục

1. **Không được để chữ / công thức đè lên nhau.** Luôn kiểm tra `.next_to()`, `.shift()`, `.to_edge()` để đảm bảo khoảng cách tối thiểu `0.3` đơn vị Manim giữa các object.
2. **Mỗi slide/scene chỉ hiển thị tối đa 1 công thức chính + 1–2 dòng chú thích.** Nếu cần nhiều hơn, chia thành nhiều bước animation.
3. **Sử dụng `VGroup` để nhóm các object liên quan** và căn chỉnh chúng cùng lúc.
4. **Luôn `FadeOut` các object cũ trước khi hiển thị nội dung mới**, trừ khi object cũ vẫn cần thiết cho ngữ cảnh.
5. **Chuyển cảnh:** Dùng `FadeOut(Group(*self.mobjects))` để xóa toàn bộ trước khi bắt đầu scene mới. Giữa các phần lớn, thêm một slide tiêu đề ngắn.
6. **Kích thước ma trận hiển thị:** Tối đa 4×4. Nếu ma trận lớn hơn, dùng dấu `\cdots` và `\vdots`.
7. **Pause:** Sau mỗi công thức/animation quan trọng, thêm `self.wait(2)` để người xem kịp đọc.

### Cấu trúc file

- Mỗi **phần lớn** (Part) là một class `Scene` riêng biệt.
- Đặt tên class theo format: `Scene01_Intro`, `Scene02_GramSchmidt`, v.v.
- Cuối cùng, tạo một class `FullVideo` render tất cả các scene liên tiếp bằng cách gọi các hàm con.

### Ma trận ví dụ xuyên suốt video

Sử dụng **một ma trận A cố định** xuyên suốt toàn bộ video để người xem theo dõi nhất quán:

```
A = [[4, 0],
     [3, -5]]
```

Ma trận này đủ nhỏ để minh họa rõ ràng trong 2D, có eigenvalue thực, và cho kết quả QR/SVD có ý nghĩa hình học.

Khi cần minh họa 3D (Gram–Schmidt), sử dụng ma trận bổ sung:

```
A_3d = [[1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]]
```

---

## PHẦN 0: MỞ ĐẦU (Title & Roadmap)
**Thời lượng:** ~1.5 phút  
**Class:** `Scene00_TitleAndRoadmap`

### Cảnh 0.1 — Title Card

**Mục đích:** Gây ấn tượng đầu tiên, giới thiệu chủ đề.

**Bố cục:**
- Toàn bộ text căn giữa màn hình theo cả 2 trục.
- Dòng 1 (title chính) ở trên, dòng 2 (subtitle) ở dưới, cách nhau `0.5` đơn vị.

**Nội dung hiển thị:**
```
Dòng 1 (font_size=56, color=WHITE, weight=BOLD):
    "Phân Rã Ma Trận"

Dòng 2 (font_size=36, color=#AAAAAA):
    "QR Decomposition  ·  Singular Value Decomposition"
```

**Animation:**
1. `FadeIn(title, shift=UP*0.3)` — `run_time=1.0`
2. `self.wait(0.5)`
3. `FadeIn(subtitle, shift=UP*0.3)` — `run_time=1.0`
4. `self.wait(2.0)`
5. `FadeOut(VGroup(title, subtitle))` — `run_time=0.8`

---

### Cảnh 0.2 — Roadmap (Lộ trình video)

**Mục đích:** Cho người xem biết video sẽ đi qua những gì.

**Bố cục:**
- Tiêu đề "Nội dung" ở phía trên (`to_edge(UP)`, `buff=0.7`).
- Danh sách 5 mục, mỗi mục là một dòng text, xếp dọc, căn trái.
- Mỗi mục có bullet number (1., 2., ...) màu `#58A6FF`, text nội dung màu `WHITE`.

**Nội dung hiển thị:**
```
Tiêu đề (font_size=44, color=WHITE):
    "Nội dung"

Các mục (font_size=32, mỗi mục cách nhau 0.5 đơn vị):
    1. Giới thiệu bài toán phân rã ma trận
    2. Phân rã QR — Gram–Schmidt
    3. Phân rã SVD — Rotate · Scale · Rotate
    4. Chéo hóa ma trận — Eigendecomposition
    5. Ứng dụng: Xấp xỉ hạng thấp (Low-Rank Approximation)
```

**Animation:**
1. `Write(title)` — `run_time=0.8`
2. Lần lượt `FadeIn(item_i, shift=RIGHT*0.3)` cho mỗi mục — `run_time=0.5`, `lag_ratio=0.3`
3. `self.wait(3.0)`
4. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

## PHẦN 1: GIỚI THIỆU BÀI TOÁN PHÂN RÃ
**Thời lượng:** ~2.5 phút  
**Class:** `Scene01_ProblemIntro`

### Cảnh 1.1 — Đặt vấn đề: Tại sao cần phân rã?

**Mục đích:** Tạo motivation — phân rã ma trận giúp hiểu sâu hơn về phép biến đổi tuyến tính.

**Bố cục:**
- Tiêu đề mục ở trên cùng, căn trái.
- Text giải thích ở giữa, căn trái, `width` tối đa `10` đơn vị Manim.

**Nội dung hiển thị:**
```
Tiêu đề (font_size=44, color=#58A6FF):
    "Tại sao cần phân rã ma trận?"

Đoạn text 1 (font_size=30, color=WHITE):
    "Một ma trận A mô tả một phép biến đổi tuyến tính."

Đoạn text 2 (font_size=30, color=WHITE):
    "Phân rã = tách A thành tích các ma trận đơn giản hơn,"
    "mỗi thành phần có ý nghĩa hình học rõ ràng."

Đoạn text 3 (font_size=30, color=#AAAAAA):
    "Giống như phân tích một lực thành các thành phần vuông góc."
```

**Animation:**
1. `Write(title)` — `run_time=0.8`
2. `self.wait(0.5)`
3. `FadeIn(text1)` — `run_time=0.8`
4. `self.wait(1.5)`
5. `FadeIn(text2)` — `run_time=0.8`
6. `self.wait(2.0)`
7. `FadeIn(text3)` — `run_time=0.8`
8. `self.wait(2.0)`
9. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 1.2 — Giới thiệu ma trận A cụ thể

**Mục đích:** Hiển thị ma trận ví dụ sẽ được dùng xuyên suốt video. Nêu rõ bài toán phân rã cần thực hiện.

**Bố cục:**
- Tiêu đề ở trên, ma trận A ở giữa-trái, phần giải thích ở giữa-phải.

**Nội dung hiển thị:**
```
Tiêu đề (font_size=40, color=WHITE):
    "Ma trận ví dụ"

Ma trận (MathTex, font_size mặc định, color=WHITE):
    A = \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix}

Chú thích bên phải (font_size=28, color=#AAAAAA):
    "Ma trận 2×2, full rank"
    "→ Tồn tại QR, SVD, Eigendecomposition"
```

**Animation:**
1. `Write(title)` — `run_time=0.8`
2. `self.wait(0.3)`
3. `Write(matrix_A)` — `run_time=1.2`
4. `self.wait(1.0)`
5. `FadeIn(annotation)` — `run_time=0.8`
6. `self.wait(2.0)`
7. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 1.3 — Tổng quan các phương pháp phân rã

**Mục đích:** So sánh nhanh QR, SVD, Eigendecomposition — mỗi cái giải quyết vấn đề gì.

**Bố cục:**
- Bảng 3 cột, mỗi cột là một phương pháp.
- Header row màu `#58A6FF`, body row màu `WHITE`.
- Sử dụng `MobjectTable` hoặc tự dựng bằng `VGroup` + `Rectangle` + `Text`.

**Nội dung bảng:**
```
| Phương pháp       | Dạng phân rã            | Ý nghĩa                     |
|-------------------|-------------------------|------------------------------|
| QR                | A = QR                  | Trực chuẩn hóa cột          |
| SVD               | A = UΣVᵀ               | Rotate → Scale → Rotate     |
| Eigendecomposition| A = PDP⁻¹              | Hướng bất biến + co giãn    |
```

**Animation:**
1. Hiển thị header row trước — `FadeIn`, `run_time=0.8`
2. Lần lượt hiển thị từng hàng — `FadeIn(row, shift=DOWN*0.2)`, `run_time=0.6` mỗi hàng
3. `self.wait(4.0)` — cho người xem đọc bảng
4. Highlight hàng "QR" bằng cách đổi màu nền sang `#58A6FF` với opacity `0.2` — `run_time=0.5`
5. Text phụ: "Bắt đầu với QR..." — `FadeIn`, `font_size=28`, `color=#AAAAAA`
6. `self.wait(1.5)`
7. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

## PHẦN 2: PHÂN RÃ QR — GRAM–SCHMIDT
**Thời lượng:** ~6 phút  
**Class:** `Scene02_QR_Decomposition`

### Cảnh 2.1 — Tiêu đề phần QR

**Bố cục:** Căn giữa màn hình.

**Nội dung:**
```
Dòng 1 (font_size=48, color=#58A6FF, weight=BOLD):
    "Phân Rã QR"

Dòng 2 (font_size=32, color=#AAAAAA):
    "Gram–Schmidt Orthogonalization"
```

**Animation:**
1. `FadeIn(title, shift=UP*0.3)` — `run_time=0.8`
2. `FadeIn(subtitle, shift=UP*0.3)` — `run_time=0.8`
3. `self.wait(1.5)`
4. `FadeOut(VGroup(title, subtitle))` — `run_time=0.8`

---

### Cảnh 2.2 — Ý tưởng QR bằng lời

**Mục đích:** Giải thích bằng lời trước khi vào công thức.

**Bố cục:** Text căn trái, bullet list.

**Nội dung:**
```
Tiêu đề (font_size=40, color=WHITE):
    "Ý tưởng cốt lõi"

Bullet 1 (font_size=30, color=WHITE):
    "• Cho ma trận A có các cột a₁, a₂, ..., aₙ"

Bullet 2 (font_size=30, color=WHITE):
    "• Tìm hệ trực chuẩn Q = [q₁, q₂, ..., qₙ]"
    "  sao cho span{q₁,...,qₖ} = span{a₁,...,aₖ} ∀k"

Bullet 3 (font_size=30, color=#FFA657):
    "• Kết quả: A = Q · R"
    "  Q trực giao, R tam giác trên"
```

**Animation:**
1. `Write(title)` — `run_time=0.8`
2. Lần lượt `FadeIn` từng bullet với `shift=LEFT*0.3` — `run_time=0.6` mỗi cái, `self.wait(1.5)` giữa các bullet
3. `self.wait(2.5)`
4. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 2.3 — Công thức Gram–Schmidt

**Mục đích:** Hiển thị công thức toán học của quá trình Gram–Schmidt.

**Bố cục:**
- Tiêu đề trên cùng.
- Công thức chiếu ở giữa trên.
- Công thức trực chuẩn hóa ở giữa dưới.

**Nội dung:**
```
Tiêu đề (font_size=40, color=WHITE):
    "Quá trình Gram–Schmidt"

Công thức 1 — Phép chiếu (MathTex, color=WHITE):
    \text{proj}_{q_j}(a_k) = \frac{\langle a_k, q_j \rangle}{\langle q_j, q_j \rangle} q_j

Chú thích 1 (font_size=26, color=#AAAAAA):
    "Chiếu aₖ lên hướng qⱼ đã có"

Công thức 2 — Trực giao hóa (MathTex, color=WHITE):
    \tilde{q}_k = a_k - \sum_{j=1}^{k-1} \text{proj}_{q_j}(a_k)

Chú thích 2 (font_size=26, color=#AAAAAA):
    "Trừ đi tất cả các thành phần đã chiếu → phần còn lại vuông góc"

Công thức 3 — Chuẩn hóa (MathTex, color=#FFA657):
    q_k = \frac{\tilde{q}_k}{\|\tilde{q}_k\|}
```

**Animation:**
1. `Write(title)` — `run_time=0.8`
2. `Write(formula1)` + `FadeIn(note1)` — `run_time=1.2`
3. `self.wait(2.0)`
4. `Write(formula2)` + `FadeIn(note2)` — `run_time=1.2`
5. `self.wait(2.0)`
6. `Write(formula3)` — `run_time=1.0`
7. `self.wait(2.5)`
8. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 2.4 — Trực quan hóa Gram–Schmidt trong 2D

**Mục đích:** Đây là **cảnh quan trọng nhất của phần QR**. Minh họa từng bước Gram–Schmidt trên không gian 2D với ma trận A = [[4,0],[3,-5]].

**Bố cục:**
- Nửa trái: Hệ tọa độ 2D (`NumberPlane` hoặc `Axes`), kích thước `[-6, 6] × [-6, 6]`, chiếm ~60% chiều rộng màn hình.
- Nửa phải: Các công thức/chú thích tương ứng với bước đang thực hiện.
- Các vector dùng `Arrow` với `stroke_width=4`.

**Màu sắc vector:**
- `a₁ = (4, 3)` → Xanh dương `#58A6FF`
- `a₂ = (0, -5)` → Cam `#FFA657`
- `q₁` (sau chuẩn hóa) → Xanh dương nhạt `#79C0FF`
- Projection vector → Đỏ nhạt `#FF7B72`, nét đứt
- `q̃₂` (phần vuông góc) → Xanh lá `#7EE787`
- `q₂` (sau chuẩn hóa) → Xanh lá nhạt `#AFFFAF`

**Các bước animation chi tiết:**

#### Bước 1: Hiển thị hệ trục và các vector cột của A
1. `Create(axes)` — `run_time=1.0`. Axes có grid lines màu `#333333`, trục x/y màu `#666666`, labels "x", "y" ở đầu trục.
2. `self.wait(0.5)`
3. `GrowArrow(a1_arrow)` — vector `a₁ = (4, 3)`, màu `#58A6FF`, label "a₁" ở đầu mũi tên — `run_time=1.0`
4. `GrowArrow(a2_arrow)` — vector `a₂ = (0, -5)`, màu `#FFA657`, label "a₂" — `run_time=1.0`
5. Text bên phải: `"Cột 1: a₁ = (4, 3)"` và `"Cột 2: a₂ = (0, -5)"` — `FadeIn`, `font_size=28`
6. `self.wait(2.0)`

#### Bước 2: Bước 1 Gram–Schmidt — q₁
1. Text bên phải thay đổi: `"Bước 1: Chuẩn hóa a₁"` — `FadeIn`
2. Công thức bên phải:
   ```
   q_1 = \frac{a_1}{\|a_1\|} = \frac{1}{5}(4, 3) = (0.8, 0.6)
   ```
3. Animation trên hệ trục:
   - Tạo bản copy của `a₁` rồi `Transform` nó thành `q₁` (vector đơn vị cùng hướng) — `run_time=1.5`
   - `q₁` hiển thị bằng màu `#79C0FF`, label "q₁"
   - Vẽ arc nhỏ (góc) từ trục x đến q₁ để thể hiện góc — `Create(arc)`, `run_time=0.5`
4. `self.wait(2.0)`

#### Bước 3: Bước 2 — Chiếu a₂ lên q₁
1. Text bên phải thay đổi: `"Bước 2: Chiếu a₂ lên q₁"` — `FadeIn`
2. Công thức:
   ```
   \text{proj}_{q_1}(a_2) = \langle a_2, q_1 \rangle \, q_1 = (-3) \cdot q_1 = (-2.4, -1.8)
   ```
3. Animation trên hệ trục:
   - Vẽ đường chấm chấm (`DashedLine`) từ đầu `a₂` vuông góc xuống hướng `q₁` — `Create`, `run_time=0.8`
   - `GrowArrow(proj_arrow)` — vector chiếu `(-2.4, -1.8)`, màu `#FF7B72`, nét đứt, label "proj" — `run_time=1.0`
   - Vẽ dấu vuông góc nhỏ (square angle indicator) tại chân đường vuông góc — `FadeIn`, `run_time=0.3`
4. `self.wait(2.0)`

#### Bước 4: Bước 2 (tiếp) — Tính q̃₂
1. Text bên phải: `"Trừ projection → phần vuông góc"` — `FadeIn`
2. Công thức:
   ```
   \tilde{q}_2 = a_2 - \text{proj}_{q_1}(a_2) = (0, -5) - (-2.4, -1.8) = (2.4, -3.2)
   ```
3. Animation trên hệ trục:
   - `GrowArrow(q2_tilde)` — vector `(2.4, -3.2)`, màu `#7EE787`, label "q̃₂" — `run_time=1.0`
   - Flash/highlight dấu vuông góc giữa `q₁` và `q̃₂` — `Indicate`, `run_time=0.5`
4. `self.wait(2.0)`

#### Bước 5: Chuẩn hóa q̃₂ → q₂
1. Text bên phải: `"Chuẩn hóa q̃₂ → q₂"` — `FadeIn`
2. Công thức:
   ```
   q_2 = \frac{\tilde{q}_2}{\|\tilde{q}_2\|} = (0.6, -0.8)
   ```
3. Animation:
   - `Transform(q2_tilde_copy, q2)` — thu nhỏ vector thành đơn vị — `run_time=1.5`
   - `q₂` màu `#AFFFAF`, label "q₂"
4. `self.wait(2.0)`

#### Bước 6: Hiển thị kết quả — hệ trực chuẩn {q₁, q₂}
1. `FadeOut` tất cả vector cũ (`a₁`, `a₂`, projection, `q̃₂`), chỉ giữ lại `q₁` và `q₂` trên hệ trục.
2. Vẽ đường tròn đơn vị (`Circle(radius=1)`, nét mờ `stroke_opacity=0.3`, màu `#AAAAAA`) để thấy q₁, q₂ nằm trên circle.
3. Text bên phải: `"q₁ ⊥ q₂,   ‖q₁‖ = ‖q₂‖ = 1"` — `FadeIn`, `color=#7EE787`
4. `self.wait(2.5)`

---

### Cảnh 2.5 — Trực quan hóa Gram–Schmidt trong 3D

**Mục đích:** Mở rộng trực quan sang không gian 3D để khẳng định tính tổng quát của phương pháp.

**Bố cục:**
- Toàn màn hình dùng `ThreeDScene` với `ThreeDAxes`.
- Camera xoay nhẹ để cho góc nhìn phối cảnh.

**Ma trận ví dụ 3D:**
```
A = [[1, 1, 0],
     [1, 0, 1],
     [0, 1, 1]]
```

**Các bước animation:**

1. **Hiển thị 3 vector cột** `a₁ = (1,1,0)`, `a₂ = (1,0,1)`, `a₃ = (0,1,1)`:
   - Dùng `Arrow3D` hoặc `Line3D` + đầu mũi tên.
   - Màu lần lượt: `#58A6FF`, `#FFA657`, `#D2A8FF`.
   - Labels 3D đặt gần đầu mũi tên.
   - `run_time=1.0` cho mỗi vector.

2. **Camera rotation** — `self.move_camera(phi=70*DEGREES, theta=-45*DEGREES)` — `run_time=2.0`

3. **Bước 1: q₁** — chuẩn hóa a₁:
   - Transform a₁ → q₁ (vector đơn vị) — `run_time=1.5`

4. **Bước 2: q₂** — chiếu a₂ lên q₁, trừ đi, chuẩn hóa:
   - Vẽ mặt phẳng span{q₁} (plane mỏng, opacity thấp) — `FadeIn`, `run_time=0.5`
   - Vẽ đường chiếu từ a₂ xuống mặt phẳng — `DashedLine` — `run_time=0.8`
   - GrowArrow q₂ — `run_time=1.0`

5. **Bước 3: q₃** — chiếu a₃ lên span{q₁, q₂}, trừ đi, chuẩn hóa:
   - Vẽ mặt phẳng span{q₁, q₂} (surface mỏng, opacity thấp, màu `#58A6FF` nhạt) — `FadeIn`, `run_time=0.8`
   - Vẽ đường chiếu từ a₃ xuống mặt phẳng — `DashedLine` — `run_time=0.8`
   - GrowArrow q₃ — hướng vuông góc với mặt phẳng — `run_time=1.0`

6. **Kết quả:** Xoay camera chậm (`run_time=3.0`) để thấy 3 vector q₁, q₂, q₃ vuông góc đôi một, tạo thành hệ tọa độ mới.

7. `self.wait(2.0)`
8. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 2.6 — Xây dựng ma trận Q và R

**Mục đích:** Từ kết quả Gram–Schmidt, ghép thành Q và R, hiển thị A = QR.

**Bố cục:**
- Công thức căn giữa màn hình.
- Chia thành 3 bước nhỏ.

**Nội dung:**

**Bước 1:** Hiển thị Q
```latex
Q = \begin{pmatrix} q_1 & q_2 \end{pmatrix} = \begin{pmatrix} 0.8 & 0.6 \\ 0.6 & -0.8 \end{pmatrix}
```
- Chú thích: `"Q là ma trận trực giao: QᵀQ = I"` — `font_size=28`, `color=#AAAAAA`

**Bước 2:** Hiển thị R
```latex
R = \begin{pmatrix} \langle a_1, q_1 \rangle & \langle a_2, q_1 \rangle \\ 0 & \langle a_2, q_2 \rangle \end{pmatrix} = \begin{pmatrix} 5 & -3 \\ 0 & 4 \end{pmatrix}
```
- Chú thích: `"R là tam giác trên: phần tử dưới đường chéo = 0"` — `font_size=28`, `color=#AAAAAA`
- Highlight các phần tử 0 ở dưới đường chéo bằng `SurroundingRectangle`, màu `#FF7B72`

**Bước 3:** Kiểm chứng A = QR
```latex
A = QR = \begin{pmatrix} 0.8 & 0.6 \\ 0.6 & -0.8 \end{pmatrix} \begin{pmatrix} 5 & -3 \\ 0 & 4 \end{pmatrix} = \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix} \checkmark
```
- Dấu check ✓ màu `#7EE787`, xuất hiện cuối cùng với hiệu ứng `FadeIn(scale=1.5)`.

**Animation:**
1. `Write(Q_formula)` — `run_time=1.2`
2. `FadeIn(Q_note)` — `run_time=0.5`
3. `self.wait(2.0)`
4. `TransformMatchingTex` hoặc `ReplacementTransform` để chuyển sang R — `run_time=1.5`
5. `self.wait(2.0)`
6. Hiển thị phép nhân QR = A — `Write`, `run_time=2.0`
7. `FadeIn(checkmark)` — `run_time=0.5`
8. `self.wait(2.5)`
9. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 2.7 — Tính chất quan trọng của Q

**Mục đích:** Nhấn mạnh tính chất trực giao — minh họa bằng phép nhân QᵀQ.

**Bố cục:** Công thức căn giữa.

**Nội dung:**
```latex
Q^T Q = \begin{pmatrix} 0.8 & 0.6 \\ 0.6 & -0.8 \end{pmatrix}^T \begin{pmatrix} 0.8 & 0.6 \\ 0.6 & -0.8 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I
```

Chú thích: `"Q trực giao ⟹ Q⁻¹ = Qᵀ  (rất tiện tính toán!)"` — `color=#FFA657`

**Animation:**
1. `Write(formula)` — `run_time=1.5`
2. `self.wait(1.5)`
3. `FadeIn(note)` — `run_time=0.8`
4. `self.wait(2.5)`
5. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

## PHẦN 3: PHÂN RÃ SVD
**Thời lượng:** ~7 phút  
**Class:** `Scene03_SVD_Decomposition`

### Cảnh 3.1 — Tiêu đề phần SVD

**Bố cục:** Căn giữa.

**Nội dung:**
```
Dòng 1 (font_size=48, color=#FFA657, weight=BOLD):
    "Phân Rã SVD"

Dòng 2 (font_size=32, color=#AAAAAA):
    "Singular Value Decomposition"

Dòng 3 (font_size=28, color=#AAAAAA):
    "A = UΣVᵀ  —  Rotate · Scale · Rotate"
```

**Animation:**
1. `FadeIn` lần lượt 3 dòng — `run_time=0.8` mỗi dòng, `self.wait(0.5)` giữa
2. `self.wait(2.0)`
3. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 3.2 — Ý tưởng hình học của SVD

**Mục đích:** **Cảnh quan trọng nhất** — SVD = xoay → co giãn → xoay. Minh họa bằng hình tròn đơn vị biến thành ellipse.

**Bố cục:**
- Toàn màn hình là hệ trục 2D.
- Hệ trục: `Axes(x_range=[-7,7], y_range=[-7,7])`, grid nhạt.

**Quy trình animation:**

#### Phase 1: Hình tròn đơn vị ban đầu
1. Vẽ hệ trục — `Create(axes)`, `run_time=0.8`
2. Vẽ hình tròn đơn vị `Circle(radius=1)` tâm gốc, màu `#58A6FF`, `stroke_width=3` — `Create`, `run_time=1.0`
3. Đánh dấu 8–12 điểm trên đường tròn bằng `Dot(radius=0.05)` màu trắng, cách đều nhau — `FadeIn`, `run_time=0.5`
4. Vẽ 2 vector cơ sở: `e₁ = (1,0)` màu `#58A6FF`, `e₂ = (0,1)` màu `#FFA657` — `GrowArrow`, `run_time=0.8` mỗi cái
5. Label text ở trên: `"Trước biến đổi: hình tròn đơn vị"` — `FadeIn`, `font_size=30`, `color=#AAAAAA`
6. `self.wait(2.0)`

#### Phase 2: Áp dụng phép biến đổi A → Ellipse
1. `FadeOut(label)`
2. Label mới: `"Nhân với A: x ↦ Ax"` — `FadeIn`, `font_size=30`, `color=WHITE`
3. `ApplyMatrix([[4,0],[3,-5]], circle_group)` — **animation chính**, `run_time=2.5`
   - Hình tròn biến thành ellipse.
   - Các điểm đánh dấu di chuyển theo.
   - Hai vector e₁, e₂ biến thành Ae₁, Ae₂.
4. Label: `"Sau biến đổi: ellipse"` — `FadeIn`, `font_size=30`, `color=#AAAAAA`
5. `self.wait(2.0)`

#### Phase 3: Phân tích ngược — SVD
1. `FadeOut` tất cả, reset lại hình tròn ban đầu.
2. Label: `"SVD tách biến đổi thành 3 bước: Vᵀ → Σ → U"` — `FadeIn`, `font_size=28`
3. `self.wait(1.5)`

**Sub-phase 3a: Bước 1 — Xoay bởi Vᵀ**
- Tính trước: `V` từ SVD của A (sử dụng kết quả code).
  ```
  Vᵀ ≈ [[ 0.8  0.6],
         [-0.6  0.8]]   (ma trận xoay)
  ```
- Label: `"Bước 1: Xoay bởi Vᵀ"` — `color=#58A6FF`
- `ApplyMatrix(Vt, circle_group)` — `run_time=2.0`
- Hình tròn vẫn là hình tròn (vì Vᵀ trực giao → giữ hình tròn).
- Chú thích: `"Vᵀ trực giao → hình tròn vẫn là hình tròn (chỉ xoay)"` — `font_size=26`, `color=#AAAAAA`
- `self.wait(2.0)`

**Sub-phase 3b: Bước 2 — Co giãn bởi Σ**
- ```
  Σ = [[σ₁, 0],
       [0, σ₂]]    (σ₁ ≈ 6.32, σ₂ ≈ 3.16)
  ```
- Label: `"Bước 2: Co giãn bởi Σ"` — `color=#D2A8FF`
- `ApplyMatrix(Sigma, group)` — `run_time=2.0`
- Hình tròn biến thành ellipse, trục dài theo hướng x (σ₁), trục ngắn theo hướng y (σ₂).
- Hiển thị các label `"σ₁"`, `"σ₂"` dọc theo 2 trục của ellipse — `FadeIn`, `color=#D2A8FF`
- Chú thích: `"Singular values = bán trục của ellipse"` — `font_size=26`, `color=#AAAAAA`
- `self.wait(2.0)`

**Sub-phase 3c: Bước 3 — Xoay bởi U**
- ```
  U ≈ [[ 0.63 -0.77],
       [-0.77 -0.63]]   (ma trận xoay)
  ```
- Label: `"Bước 3: Xoay bởi U"` — `color=#FFA657`
- `ApplyMatrix(U, group)` — `run_time=2.0`
- Ellipse xoay về vị trí cuối — kết quả trùng với việc nhân A trực tiếp!
- `self.wait(1.0)`

#### Phase 4: Overlay so sánh
1. Hiển thị lại ellipse từ Phase 2 (nhân A trực tiếp) bằng nét đứt `DashedVMobject`, màu `#7EE787`, `stroke_opacity=0.5`.
2. Overlay lên kết quả Phase 3 — hai ellipse trùng nhau.
3. Text: `"Kết quả giống hệt nhau! A = UΣVᵀ ✓"` — `color=#7EE787`, `font_size=32`
4. `self.wait(3.0)`
5. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 3.3 — Công thức SVD chi tiết

**Mục đích:** Trình bày các bước tính toán SVD từ AᵀA.

**Bố cục:** Nửa trên và nửa dưới, mỗi phần một nhóm công thức.

**Nội dung - Bước tính:**

**Bước 1:** Tính AᵀA
```latex
A^T A = \begin{pmatrix} 4 & 3 \\ 0 & -5 \end{pmatrix} \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix} = \begin{pmatrix} 25 & -15 \\ -15 & 25 \end{pmatrix}
```

**Bước 2:** Tìm eigenvalues của AᵀA → singular values
```latex
\lambda_1 = 40, \quad \lambda_2 = 10

\sigma_1 = \sqrt{40} \approx 6.32, \quad \sigma_2 = \sqrt{10} \approx 3.16
```
- Highlight `σ₁`, `σ₂` bằng `SurroundingRectangle`, màu `#D2A8FF`

**Bước 3:** Eigenvectors → V
```latex
V = \begin{pmatrix} v_1 & v_2 \end{pmatrix}
```
- Chú thích: `"Eigenvectors của AᵀA → cột của V"`

**Bước 4:** U = AV Σ⁻¹
```latex
u_i = \frac{1}{\sigma_i} A v_i
```
- Chú thích: `"Tính U từ A, V và σ"`

**Animation:**
- Hiển thị lần lượt 4 bước, mỗi bước `FadeIn` rồi `self.wait(2.5)`.
- Giữa mỗi bước, `FadeOut` bước trước (giữ lại tiêu đề phần).
- `FadeOut(Group(*self.mobjects))` cuối cùng.

---

### Cảnh 3.4 — Kết quả SVD số cho ma trận A

**Mục đích:** Hiển thị kết quả SVD cụ thể, kiểm chứng A = UΣVᵀ.

**Bố cục:** Toàn bộ phép nhân trải ngang giữa màn hình.

**Nội dung:**
```latex
A = U \Sigma V^T

\begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix}
=
\underbrace{\begin{pmatrix} u_{11} & u_{12} \\ u_{21} & u_{22} \end{pmatrix}}_{U}
\underbrace{\begin{pmatrix} \sigma_1 & 0 \\ 0 & \sigma_2 \end{pmatrix}}_{\Sigma}
\underbrace{\begin{pmatrix} v_{11} & v_{12} \\ v_{21} & v_{22} \end{pmatrix}^T}_{V^T}
```

Sau đó thay số cụ thể (lấy từ output của `QR_SVD.py`):
```latex
= \begin{pmatrix} 0.63 & 0.77 \\ -0.77 & 0.63 \end{pmatrix}
  \begin{pmatrix} 6.32 & 0 \\ 0 & 3.16 \end{pmatrix}
  \begin{pmatrix} 0.8 & 0.6 \\ -0.6 & 0.8 \end{pmatrix}
```

Dấu `✓` check mark — `color=#7EE787`

**Animation:**
1. `Write(general_formula)` — `run_time=1.5`
2. `self.wait(2.0)`
3. `TransformMatchingTex(general → numerical)` — `run_time=2.0`
4. `self.wait(1.0)`
5. `FadeIn(checkmark)` — `run_time=0.5`
6. `self.wait(2.5)`
7. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 3.5 — Ý nghĩa hình học của Singular Values

**Mục đích:** Tổng kết ý nghĩa σ₁, σ₂ — liên hệ với ellipse.

**Bố cục:**
- Nửa trái: Ellipse minh họa với bán trục được label.
- Nửa phải: Bullet list giải thích.

**Nội dung bên phải:**
```
(font_size=28, color=WHITE):
• σ₁ = bán trục dài của ellipse
  → hướng co giãn mạnh nhất

• σ₂ = bán trục ngắn của ellipse
  → hướng co giãn yếu nhất

• Rank(A) = số singular values > 0

• σ₁/σ₂ = condition number
  → đo độ "méo" của phép biến đổi
```

**Animation:**
1. Vẽ ellipse bên trái với labels σ₁, σ₂ — `Create`, `run_time=1.0`
2. Lần lượt `FadeIn` từng bullet bên phải — `run_time=0.6` mỗi cái, `self.wait(1.5)` giữa mỗi bullet
3. `self.wait(3.0)`
4. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

## PHẦN 4: CHÉO HÓA MA TRẬN (EIGENDECOMPOSITION)
**Thời lượng:** ~4 phút  
**Class:** `Scene04_Eigendecomposition`

### Cảnh 4.1 — Tiêu đề

**Nội dung:**
```
Dòng 1 (font_size=48, color=#D2A8FF, weight=BOLD):
    "Chéo Hóa Ma Trận"

Dòng 2 (font_size=32, color=#AAAAAA):
    "A = PDP⁻¹"
```

**Animation:** Tương tự Cảnh 2.1.

---

### Cảnh 4.2 — Eigenvalue & Eigenvector là gì?

**Mục đích:** Nhắc lại khái niệm trước khi vào phân rã.

**Bố cục:** Text + hệ trục 2D minh họa.

**Nội dung text:**
```
Tiêu đề (font_size=40, color=WHITE):
    "Eigenvalue & Eigenvector"

Công thức định nghĩa (MathTex):
    Av = \lambda v

Giải thích (font_size=28, color=#AAAAAA):
    "v là eigenvector: A chỉ co giãn v, không đổi hướng"
    "λ là eigenvalue: hệ số co giãn tương ứng"
```

**Minh họa trên hệ trục (nửa dưới hoặc nửa phải):**
1. Vẽ vector `v` (eigenvector) — màu `#D2A8FF`
2. Vẽ vector `Av = λv` (cùng hướng, dài hơn/ngắn hơn) — màu `#FFA657`
3. Animation: `Transform(v_copy, Av)` — thể hiện vector giữ nguyên hướng nhưng thay đổi độ dài.

Dùng eigenvalues thực tế của A:
```
A = [[4, 0], [3, -5]]
λ₁ = 4,  v₁ = (1, -⅓) → chuẩn hóa
λ₂ = -5, v₂ = (0, 1)
```

**Animation chi tiết:**
1. `Write(definition)` — `run_time=1.0`
2. `self.wait(1.5)`
3. `FadeIn(explanation)` — `run_time=0.8`
4. `self.wait(1.0)`
5. Minh họa vector v₁:
   - `GrowArrow(v1)` — `run_time=0.8`
   - `Transform(v1_copy, Av1)` — v₁ → λ₁v₁ = 4v₁, cùng hướng, dài gấp 4 — `run_time=1.5`
   - Text: `"λ₁ = 4: co giãn gấp 4, giữ hướng"` — `FadeIn`, `color=#D2A8FF`
6. `self.wait(1.5)`
7. Minh họa vector v₂:
   - `GrowArrow(v2)` — `run_time=0.8`
   - `Transform(v2_copy, Av2)` — v₂ → λ₂v₂ = -5v₂, **đổi chiều** và dài gấp 5 — `run_time=1.5`
   - Text: `"λ₂ = -5: co giãn gấp 5, đảo chiều (λ < 0)"` — `FadeIn`, `color=#FF7B72`
8. `self.wait(2.5)`
9. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 4.3 — Xây dựng A = PDP⁻¹

**Mục đích:** Hiển thị phân rã eigendecomposition cụ thể.

**Bố cục:** Công thức căn giữa, chia 3 bước.

**Nội dung:**

**Bước 1:** Ma trận P (eigenvectors làm cột)
```latex
P = \begin{pmatrix} v_1 & v_2 \end{pmatrix}
```
- Chú thích: `"P: các eigenvector xếp thành cột"` — `color=#AAAAAA`
- Highlight các cột bằng `SurroundingRectangle`: cột 1 màu `#58A6FF`, cột 2 màu `#FFA657`

**Bước 2:** Ma trận D (eigenvalues trên đường chéo)
```latex
D = \begin{pmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{pmatrix} = \begin{pmatrix} 4 & 0 \\ 0 & -5 \end{pmatrix}
```
- Chú thích: `"D: eigenvalues trên đường chéo"` — `color=#AAAAAA`
- Highlight đường chéo bằng màu `#D2A8FF`

**Bước 3:** Kiểm chứng
```latex
A = PDP^{-1} = \begin{pmatrix} v_1 & v_2 \end{pmatrix} \begin{pmatrix} 4 & 0 \\ 0 & -5 \end{pmatrix} \begin{pmatrix} v_1 & v_2 \end{pmatrix}^{-1}
```

Thay số:
```latex
= \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix} \checkmark
```

**Animation:**
1. Hiển thị P — `Write`, `run_time=1.2`
2. `self.wait(2.0)`
3. Transform sang D — `run_time=1.5`
4. `self.wait(2.0)`
5. Hiển thị phép nhân PDP⁻¹ = A — `Write`, `run_time=2.0`
6. `FadeIn(checkmark)` — `run_time=0.5`
7. `self.wait(2.5)`
8. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 4.4 — Ý nghĩa hình học: Đổi hệ tọa độ → Co giãn → Đổi lại

**Mục đích:** Trực quan hóa ý nghĩa PDP⁻¹ = đổi cơ sở → co giãn theo eigen-directions → đổi lại.

**Bố cục:** Hệ trục 2D toàn màn hình.

**Animation:**

1. **Hệ tọa độ chuẩn** — vẽ e₁, e₂ (trục standard) và hình vuông đơn vị (4 đỉnh).

2. **Bước 1: P⁻¹ — đổi cơ sở sang eigenvector**
   - Label: `"P⁻¹: nhìn trong hệ eigenvector"` — `color=#58A6FF`
   - Vẽ các eigenvector v₁, v₂ làm trục mới (nét đứt) — `Create`, `run_time=0.8`
   - `ApplyMatrix(P_inv, square_group)` — biến đổi hình vuông — `run_time=2.0`
   - `self.wait(1.5)`

3. **Bước 2: D — co giãn theo eigenvalues**
   - Label: `"D: co giãn theo λ₁ = 4, λ₂ = −5"` — `color=#D2A8FF`
   - `ApplyMatrix(D, group)` — `run_time=2.0`
   - `self.wait(1.5)`

4. **Bước 3: P — đổi lại hệ tọa độ chuẩn**
   - Label: `"P: quay về hệ chuẩn"` — `color=#FFA657`
   - `ApplyMatrix(P, group)` — `run_time=2.0`
   - `self.wait(1.5)`

5. Text kết luận: `"PDP⁻¹ = A ⟹ A chỉ co giãn theo các hướng eigenvector"` — `FadeIn`, `color=#7EE787`, `font_size=28`
6. `self.wait(3.0)`
7. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

## PHẦN 5: ỨNG DỤNG — XẤP XỈ HẠNG THẤP (LOW-RANK APPROXIMATION)
**Thời lượng:** ~3 phút  
**Class:** `Scene05_LowRankApprox`

### Cảnh 5.1 — Tiêu đề

**Nội dung:**
```
Dòng 1 (font_size=44, color=#7EE787, weight=BOLD):
    "Ứng dụng SVD"

Dòng 2 (font_size=32, color=#AAAAAA):
    "Xấp xỉ hạng thấp — Low-Rank Approximation"
```

**Animation:** FadeIn lần lượt → wait → FadeOut.

---

### Cảnh 5.2 — Công thức xấp xỉ hạng thấp

**Mục đích:** Giải thích kỹ thuật cắt bỏ singular values nhỏ.

**Bố cục:** Công thức + chú thích.

**Nội dung:**
```latex
A \approx A_k = \sum_{i=1}^{k} \sigma_i \, u_i \, v_i^T
```

Chú thích:
```
"Giữ lại k singular values lớn nhất"
"→ Ma trận A_k có rank = k"
"→ Xấp xỉ tốt nhất theo Frobenius norm (Eckart-Young theorem)"
```

**Animation:**
1. `Write(formula)` — `run_time=1.2`
2. Lần lượt `FadeIn` chú thích — `run_time=0.6` mỗi dòng
3. `self.wait(3.0)`
4. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 5.3 — Minh họa trực quan: Giảm rank

**Mục đích:** Dùng ma trận lớn hơn (hoặc ảnh grayscale) để minh họa hiệu ứng của giảm rank.

**Lựa chọn A:** Có thể dùng ma trận 4×4 hoặc ảnh pixel đơn giản.

**Nếu dùng ma trận 4×4:**
```
A_demo = [[3, 2, 2, 1],
          [2, 3, 1, 2],
          [2, 1, 3, 2],
          [1, 2, 2, 3]]
```

**Bố cục:**
- Hiển thị ma trận A gốc bên trái.
- Hiển thị A₁ (rank-1), A₂ (rank-2), A₃ (rank-3) lần lượt bên phải.
- Hiển thị sai số ‖A − Aₖ‖ cho mỗi k.

**Animation:**
1. `Write(A_matrix)` — `run_time=1.0`
2. `self.wait(1.0)`
3. Highlight σ₁ → tạo A₁ — `Transform`, `run_time=1.5`
4. Hiện sai số: `"‖A − A₁‖ = ..."` — `FadeIn`, `color=#FF7B72`
5. `self.wait(1.5)`
6. Thêm σ₂ → tạo A₂ — `Transform`, `run_time=1.5`
7. Hiện sai số giảm đi — `FadeIn`, `color=#FFA657`
8. `self.wait(1.5)`
9. Thêm σ₃ → tạo A₃ — `Transform`, `run_time=1.5`
10. Sai số rất nhỏ — `FadeIn`, `color=#7EE787`
11. `self.wait(1.5)`
12. Text kết luận: `"Càng nhiều singular values → xấp xỉ càng chính xác"` — `FadeIn`, `color=#7EE787`
13. `self.wait(2.5)`
14. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 5.4 — Bar chart: Singular Values giảm dần

**Mục đích:** Minh họa trực quan sự phân bố singular values — thường giảm rất nhanh, nên chỉ cần vài giá trị lớn.

**Bố cục:** `BarChart` căn giữa màn hình.

**Nội dung:**
- Trục x: "σ₁, σ₂, σ₃, σ₄"
- Trục y: Giá trị singular values
- Bars: Gradient từ `#D2A8FF` (σ₁) đến `#444444` (σ₄)
- Đường ngang tại threshold — `DashedLine`, `color=#FF7B72`, label "cutoff"
- Bars dưới threshold → mờ đi (opacity giảm) → "bỏ qua"

**Animation:**
1. `Create(chart)` — `run_time=1.5`
2. `self.wait(1.0)`
3. `Create(threshold_line)` — `run_time=0.5`
4. Bars dưới threshold → `animate.set_opacity(0.2)` — `run_time=1.0`
5. Text: `"Giữ σ lớn, bỏ σ nhỏ → nén dữ liệu!"` — `FadeIn`, `color=#AAAAAA`
6. `self.wait(3.0)`
7. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

## PHẦN 6: TỔNG KẾT VÀ OUTRO
**Thời lượng:** ~1.5 phút  
**Class:** `Scene06_Summary`

### Cảnh 6.1 — Bảng tổng kết

**Bố cục:** Bảng so sánh 3 phương pháp, căn giữa.

**Nội dung bảng:**
```
| Phương pháp  | Dạng           | Điều kiện     | Ứng dụng chính              |
|--------------|----------------|---------------|------------------------------|
| QR           | A = QR         | Full rank     | Giải hệ, least squares      |
| SVD          | A = UΣVᵀ      | Mọi ma trận   | Nén, PCA, pseudo-inverse     |
| Eigen        | A = PDP⁻¹     | Square, diag. | Lũy thừa ma trận, ODE       |
```

**Màu header:** `#58A6FF`
**Màu hàng:** Xen kẽ `#2a2a2a` / `#1e1e1e` (stripe)

**Animation:**
1. `FadeIn(table)` — `run_time=1.5`
2. `self.wait(5.0)`
3. `FadeOut(table)` — `run_time=0.8`

---

### Cảnh 6.2 — Liên hệ giữa 3 phương pháp

**Mục đích:** Vẽ sơ đồ mối quan hệ.

**Bố cục:** 3 box nối bằng mũi tên, dạng flowchart đơn giản.

**Nội dung:**
```
[QR] ──"QR iteration"──→ [Eigenvalues]
  │                            │
  │                            ↓
  └──"AᵀA eigen"────→ [SVD]
```

Mỗi box: `RoundedRectangle` + `Text` bên trong.
- QR box: viền `#58A6FF`
- Eigen box: viền `#D2A8FF`
- SVD box: viền `#FFA657`

**Animation:**
1. `FadeIn` 3 boxes lần lượt — `run_time=0.5` mỗi cái
2. `Create` arrows giữa chúng — `run_time=0.5` mỗi arrow
3. `FadeIn` labels trên arrows — `run_time=0.3` mỗi label
4. `self.wait(4.0)`
5. `FadeOut(Group(*self.mobjects))` — `run_time=0.8`

---

### Cảnh 6.3 — Outro

**Bố cục:** Căn giữa.

**Nội dung:**
```
Dòng 1 (font_size=44, color=WHITE):
    "Cảm ơn đã theo dõi!"

Dòng 2 (font_size=30, color=#AAAAAA):
    "MatrixLab — HCMUS"

Dòng 3 (font_size=24, color=#666666):
    "Applied Mathematics • 2026"
```

**Animation:**
1. `FadeIn(line1)` — `run_time=1.0`
2. `self.wait(0.5)`
3. `FadeIn(line2)` — `run_time=0.8`
4. `FadeIn(line3)` — `run_time=0.8`
5. `self.wait(3.0)`
6. `FadeOut(Group(*self.mobjects))` — `run_time=1.0`

---

## PHỤ LỤC: SỐ LIỆU CỤ THỂ ĐỂ HARD-CODE

### QR Decomposition của A = [[4,0],[3,-5]]

```python
# Kết quả từ QR_SVD.py
Q = [[ 0.8,  0.6],
     [ 0.6, -0.8]]

R = [[ 5.0, -3.0],
     [ 0.0,  4.0]]

# Kiểm chứng: QR = [[4,0],[3,-5]] ✓
```

### SVD của A = [[4,0],[3,-5]]

```python
# Kết quả từ QR_SVD.py
U = [[ 0.6325,  0.7746],
     [-0.7746,  0.6325]]

Sigma = [[ 6.3246, 0     ],
         [ 0,      3.1623]]

Vt = [[ 0.8, 0.6],
      [-0.6, 0.8]]

# Kiểm chứng: U @ Sigma @ Vt = [[4,0],[3,-5]] ✓
```

### Eigendecomposition của A = [[4,0],[3,-5]]

```python
# Eigenvalues
lambda_1 = 4
lambda_2 = -5

# Eigenvectors (cột)
# Av = λv
# A @ [1, -1/3] = 4 * [1, -1/3]  →  v₁ = [1, -1/3] (chuẩn hóa)
# A @ [0, 1]    = -5 * [0, 1]    →  v₂ = [0, 1]

# Chú ý: eigenvector có thể cần chuẩn hóa tùy convention
P = [[ 1,    0],
     [-1/3,  1]]   # hoặc chuẩn hóa

D = [[ 4,  0],
     [ 0, -5]]

# Kiểm chứng: P @ D @ P_inv = A ✓
```

---

## CHECKLIST CHẤT LƯỢNG

Trước khi submit code Manim, AI cần kiểm tra:

- [ ] **Không có chữ đè nhau.** Chạy thử và kiểm tra mọi frame.
- [ ] **Font size phù hợp.** Heading: 40–48, body: 28–36, chú thích: 24–28.
- [ ] **Nền xám `#1e1e1e`, chữ trắng `#FFFFFF`.** Không dùng nền trắng, không dùng chữ đen.
- [ ] **Mỗi scene clear hết trước khi sang scene mới** (`FadeOut(Group(*self.mobjects))`).
- [ ] **Thời gian wait đủ dài** để đọc (~2–3 giây cho mỗi công thức).
- [ ] **Tổng thời lượng 15–25 phút.** Tính toán: ~120–150 slides × ~8–10 giây/slide.
- [ ] **Ma trận ví dụ nhất quán** — luôn dùng A = [[4,0],[3,-5]] trừ khi cần minh họa khác.
- [ ] **Màu sắc nhất quán** — tuân theo bảng color palette ở đầu file.
- [ ] **Code chạy được** — không import thiếu, không syntax error.
- [ ] **Có `self.wait()` sau mỗi animation quan trọng.**

# Họ và tên: Đỗ Bá Luân.

Mssv: 23520895.

Môn: DS102.R11.CNVN.2 - Thực hành.

---
### Tóm tắt nội dung bài nộp:

**1. Thư mục BTTH1:** 
- Em đã hoàn thành xong phần cào dữ liệu và xuất ra đủ các định dạng (csv, json, tsv).

**2. Thư mục BTTH2 (Logistic & Softmax Regression):**
- **Dữ liệu MNIST (Thầy cấp):** Do dữ liệu em tự crawl ở BTTH1 chủ yếu là text và thông tin sản phẩm, không phải dạng ma trận ảnh chuẩn như yêu cầu của bài thực hành, nên em dùng tập MNIST của thầy để làm trọn vẹn phần chính (code tay thuật toán Logistic & Softmax Regression).
- **Dữ liệu tự crawl (Phần Bonus):** Dùng file data của BTTH1, ở file `assignment_bonus.py`, em đã tự trích xuất thêm đặc trưng (giá tiền, độ dài tên sản phẩm) từ file `data.json` để chạy thử trên model Logistic Regression vừa code ở trên. Mục đích là để kiếm điểm cộng và test xem thuật toán mình tự code chạy trên dữ liệu thực tế tự cào sẽ ra sao.

*(Tất cả biểu đồ hàm loss và ma trận nhầm lẫn em đã xuất ảnh lưu sẵn trong thư mục `figures/` để thầy tiện xem lướt qua).*

**3. Thư mục BTTH3 (Linear Regression):**
- **Dữ liệu Forest Fires (Thầy cấp):** Sử dụng bộ dữ liệu Forest Fires từ UCI theo đúng yêu cầu đề bài để thực hiện trọn vẹn 4 Assignment:
  - `assignment_1.py`: Cài đặt mô hình Linear Regression giải tích OLS thuần bằng thư viện NumPy.
  - `assignment_2.py`: Chuẩn hóa Z-Score trên tập Train rồi áp dụng sang Test (tránh Data Leakage); thêm cột bias sau khi chuẩn hóa để tránh chia cho 0; so sánh với kết quả gốc.
  - `assignment_3.py`: Vẽ Heatmap ma trận tương quan giữa các đặc trưng (lưu tại `figures/correlation_heatmap.png`), phát hiện tương quan đa cộng tuyến mạnh giữa `DC` và `DMC` (khoảng  0.68), loại bỏ cột `DC` giúp cải thiện mô hình (MSE giảm từ 571.25 xuống 557.12).
  - `assignment_4.py`: Đối chiếu với thư viện Scikit-Learn `LinearRegression`, kết quả MSE và R^2 trùng khớp hoàn toàn với mô hình.
- **Dữ liệu tự crawl (Phần Bonus - `assignment_bonus.py`):** 
  - Dùng dữ liệu đã cào ở BTTH1 (`data.json`), em trích xuất các thuộc tính thực tế của sản phẩm (`title_length`, `word_count`, phân loại mặt hàng `is_music`, `is_clothing`, `is_bundle`) để xây dựng bài toán hồi quy dự đoán giá bán (`price`).
  - Mô hình OLS tự code bằng NumPy thuần đạt R^2 khoảng 52.23 trên tập Test, biểu đồ phân tán so sánh giá thực tế vs giá dự đoán được lưu tại `figures/bonus_regression.png`.


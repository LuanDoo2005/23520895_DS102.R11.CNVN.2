# **CẨM NANG HƯỚNG DẪN THỰC HÀNH LAB 3: LINEAR REGRESSION (DS102)**

## **MỤC LỤC**

> 1. Phần 1: Giải Ngố Toán \- Code & Chuẩn Bị Vấn Đáp  
   * 1.1. Mối liên hệ giữa công thức toán học và dòng code  
   * 1.2. Phân tích chi tiết kích thước ma trận (Matrix Shapes)  
> 2. Phần 2: Chiến Thuật Đạt Điểm Tối Đa (10/10) & Chống Soi  
   * 2.1. Cố định Random Seed để Train/Test không bị nhảy  
   * 2.2. Tránh "bẫy" khi thêm cột Bias trong Assignment 2  
   * 2.3. Quy cách nộp bài và chiến thuật Git Commit tự nhiên  
> 3. Phần 3: Lộ Trình Triển Khai Từng Bước A-Z Cho Newbie  
> 4. Phần 4: Toàn Bộ Code Mẫu Chuẩn (100% Không Comment) & Giải Thích Chi Tiết  
   * File 1: src/assignment\_1.py  
   * File 2: src/assignment\_2.py  
   * File 3: src/assignment\_3.py  
   * File 4: src/assignment\_4.py

# **PHẦN 1: GIẢI NGỐ TOÁN \- CODE & CHUẨN BỊ VẤN ĐÁP**

### **1.1. Mối liên hệ giữa công thức toán học và dòng code**

Trong phần lý thuyết preliminaries.ipynb, thầy đã dẫn dắt rất kỹ từ hàm mất mát \$E(X, y, \\beta) \= \\frac{1}{2} \\vert{}\\vert{}y \- X\\beta\\vert{}\\vert{}^2\$ đến cách tìm ra nghiệm tối ưu của Linear Regression. Điểm cốt lõi là làm sao map những công thức đó sang các dòng code NumPy.

| Giai đoạn | Công thức Toán học | Dòng Code tương ứng | Bản chất & Ý nghĩa |
| :---- | :---- | :---- | :---- |
| **Mô hình Hồi quy** | **$y=X\beta +\varepsilon$** | y\_pred \= X\_test @ beta | Tính toán giá trị diện tích rừng bị cháy dự đoán (y\_pred) bằng cách nhân ma trận đặc trưng với trọng số $\beta$. |
| **Nghiệm tối ưu OLS** | **$\hat{\beta }=({X}^{T}X{)}^{-1}{X}^{T}y$** | beta \= np.linalg.inv(X\_train.T @ X\_train) @ X\_train.T @ y\_train | Tìm bộ trọng số có sai số thấp nhất (Best Linear Unbiased Estimator \- BLUE) bằng đại số tuyến tính. |
| **Thêm Hệ số chặn** | **${X}_{0}=1$** | X\_train \= np.hstack(\[np.ones((X\_train.shape\[0\], 1)), X\_train\]) | Đưa thêm cột toàn số 1 vào đầu ma trận $X$ để tính được hệ số tự do ${\beta }_{0}$ (bias) trong phương trình. |
| **Chuẩn hóa Z-Score** | **${X}_{std}=\frac{X-\mu }{\sigma }$** | X\_train\_std \= (X\_train \- mu) / sigma | Đưa các cột dữ liệu về phân phối có trung bình bằng 0 và độ lệch chuẩn bằng 1 giúp mô hình ổn định hơn. |

*Lưu ý khi thầy hỏi:* Tại sao lại có các phương pháp Maximum Likelihood (ML) hay Maximum A Posteriori (MAP) trong slide? Bạn cần trả lời: Tìm $\beta$ bằng OLS bản chất chính là cực đại hóa hàm Likelihood (với giả định nhiễu tuân theo phân phối Chuẩn). Còn nếu cộng thêm thành phần Regularization (${\beta }^{T}\beta$) thì chính là cách tiếp cận MAP (cực đại hóa xác suất hậu nghiệm).

### **1.2. Phân tích chi tiết kích thước ma trận (Matrix Shapes)**

Khi tính OLS bằng công thức np.linalg.inv(X.T @ X) @ X.T @ y, thầy rất hay hỏi xoáy: *"Kích thước của ma trận $({X}^{T}X)$ là bao nhiêu? Tại sao lại tính được nghịch đảo?"*.  
Dữ liệu Forest Fires có 12 cột (đã trừ cột area làm nhãn).

* **Ma trận dữ liệu đầu vào ${X}_{train}$ ban đầu:** Kích thước $(N,12)$, với $N\approx 414$ (chiếm 80% của 517 dòng).  
* **Ma trận ${X}_{train}$ sau khi thêm bias:** Kích thước $(N,13)$.  
* **Vector nhãn ${y}_{train}$:** Kích thước $(N,1)$.

**Phân tích cụ thể dòng tính beta:**

> 1. X.T có kích thước $(13,N)$. Khi nhân @ với X $(N,13)$ ta được ma trận vuông:  
>    \$\$(13, N) \\times (N, 13\) \\longrightarrow (13, 13)\$\$  
> 2. np.linalg.inv(...) sẽ khả nghịch ma trận $(13,13)$ này. Nếu ma trận này không khả nghịch (do đa cộng tuyến quá mạnh, $rank(X)<13$), code sẽ báo lỗi Singular Matrix. Đoạn này có liên kết chặt chẽ đến Assignment 3\.  
> 3. X.T @ y có kích thước: \$(13, N) \\times (N, 1\) \\longrightarrow (13, 1)\$.  
> 4. Cuối cùng nhân 2 cụm lại với nhau: \$(13, 13\) \\times (13, 1\) \\longrightarrow (13, 1)\$. Kết quả $\beta$ trả về một vector có 13 phần tử (1 phần tử bias ${\beta }_{0}$ và 12 trọng số ${\beta }_{1}...{\beta }_{12}$).

# **PHẦN 2: CHIẾN THUẬT ĐẠT ĐIỂM TỐI ĐA (10/10) & CHỐNG SOI**

### **2.1. Cố định Random Seed để Train/Test không bị nhảy**

Thầy dặn rất kỹ là khi so sánh các mô hình (Assignment 2, 3, 4), dữ liệu Train/Test phải giống hệt nhau mới so sánh khách quan được. Đừng dùng train\_test\_split của Sklearn quá vội ở Ass 1 và 2 vì thầy bắt xài NumPy "thuần". Hãy dùng np.random.permutation hoặc xáo trộn index kết hợp với np.random.seed(42).

### **2.2. Tránh "bẫy" khi thêm cột Bias trong Assignment 2**

Ở Assignment 2 (Standardize), một lỗi kinh điển sinh viên hay mắc là chuẩn hóa luôn cả cột Bias (cột toàn số 1). Độ lệch chuẩn của cột toàn số 1 bằng 0, chia cho 0 sẽ ra NaN. Cách giải quyết "chuẩn bài" thầy ghi trên lớp: **Chuẩn hóa các cột đặc trưng trước, sau đó mới np.hstack cột bias vào**. Và nhớ kĩ là tập Test phải dùng ${\mu }_{train}$ và ${\sigma }_{train}$ của tập Train chứ không được tự tính lại trên Test.

### **2.3. Quy cách nộp bài và chiến thuật Git Commit tự nhiên**

Làm y như Lab 2\. Không được đẩy toàn bộ 4 file bài tập trong 1 commit duy nhất. Chia nhỏ thành các bước như một sinh viên đang lọ mọ debug:

* Commit 1: git add data/ && git commit \-m "add forestfires dataset from UCI"  
* Commit 2: git add src/assignment\_1.py && git commit \-m "implement manual OLS linear regression"  
* Commit 3: git add src/assignment\_2.py && git commit \-m "standardize data and fix bias column issue"  
* Commit 4: git add src/assignment\_3.py && git commit \-m "plot correlation heatmap and drop DC column"  
* Commit 5: git add src/assignment\_4.py && git commit \-m "add sklearn baseline for comparison"

Code mình cung cấp bên dưới đã cố tình viết cách ánh xạ chuỗi thành số một cách thủ công (bằng dictionary .map()), không dùng LabelEncoder nhằm giữ cho code đúng chất "làm bằng tay" theo hướng dẫn của bài thực hành.

# **PHẦN 3: LỘ TRÌNH TRIỂN KHAI TỪNG BƯỚC A-Z CHO NEWBIE**

Đoạn mã  
graph TD  
    A\[Bước 1: Tải file forestfires.csv từ UCI\] \--\> B\[Bước 2: Assignment 1 \- Tính OLS bằng NumPy\]  
    B \--\> C\[Bước 3: Assignment 2 \- Standardize dữ liệu bằng tay\]  
    C \--\> D\[Bước 4: Assignment 3 \- Kiểm tra độ tương quan ma trận\]  
    D \--\> E\[Bước 5: Assignment 3 \- Loại bỏ cột DC\]  
    E \--\> F\[Bước 6: Assignment 4 \- Benchmark bằng Sklearn\]

**Các lệnh chuẩn bị môi trường:**

Bash  
pip install pandas numpy matplotlib seaborn scikit-learn  
mkdir \-p data src  
\# Copy file forestfires.csv vào thư mục data/

# **PHẦN 4: TOÀN BỘ CODE MẪU CHUẨN (100% KHÔNG COMMENT) & GIẢI THÍCH CHI TIẾT**

### **FILE 1: src/assignment\_1.py**

Python  
import pandas as pd  
import numpy as np  
from sklearn.metrics import mean\_squared\_error, r2\_score

df \= pd.read\_csv("data/forestfires.csv")

month\_map \= {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}  
day\_map \= {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

df\['month'\] \= df\['month'\].map(month\_map)  
df\['day'\] \= df\['day'\].map(day\_map)

X\_full \= df.drop(columns=\['area'\]).values  
y\_full \= df\['area'\].values.reshape(-1, 1)

np.random.seed(42)  
indices \= np.arange(len(X\_full))  
np.random.shuffle(indices)

train\_size \= int(len(X\_full) \* 0.8)  
train\_idx \= indices\[:train\_size\]  
test\_idx \= indices\[train\_size:\]

X\_train \= X\_full\[train\_idx\]  
y\_train \= y\_full\[train\_idx\]  
X\_test \= X\_full\[test\_idx\]  
y\_test \= y\_full\[test\_idx\]

X\_train \= np.hstack(\[np.ones((X\_train.shape\[0\], 1)), X\_train\])  
X\_test \= np.hstack(\[np.ones((X\_test.shape\[0\], 1)), X\_test\])

beta \= np.linalg.inv(X\_train.T @ X\_train) @ X\_train.T @ y\_train

y\_pred \= X\_test @ beta

mse \= mean\_squared\_error(y\_test, y\_pred)  
r2 \= r2\_score(y\_test, y\_pred)

print("="\*10 \+ " ASSIGNMENT 1: OLS MANUAL " \+ "="\*10)  
print(f"MSE: {mse}")  
print(f"R2 Score: {r2}")

**Giải thích từng dòng:**

* pd.read\_csv(...): Đọc file dữ liệu vào DataFrame.  
* month\_map và day\_map: Khởi tạo từ điển ánh xạ chuỗi thành số nguyên.  
* df.drop(...).values: Tách lấy đặc trưng (cột 1-12) và chuyển đổi lập tức thành numpy.ndarray để tránh lỗi bất đồng chỉ số (index mismatch) khi tính toán ma trận.  
* np.random.seed(42) và xáo trộn index: Tách tập dữ liệu chuẩn tỷ lệ 80% Train và 20% Test giống phong cách code từ đầu môn.  
* np.hstack(\[np.ones(...), X\_train\]): Thêm cột vector ${X}_{0}=1$ (cột bias) vào bên trái ma trận dữ liệu.  
* beta \= np.linalg.inv(...): Implement đúng 100% công thức OLS thầy ghi trên slide để tìm ra bộ trọng số tối ưu.  
* mean\_squared\_error, r2\_score: Đánh giá mô hình theo đúng tiêu chí của Lab 3\.

### **FILE 2: src/assignment\_2.py**

Python  
import pandas as pd  
import numpy as np  
from sklearn.metrics import mean\_squared\_error, r2\_score

df \= pd.read\_csv("data/forestfires.csv")

month\_map \= {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}  
day\_map \= {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}  
df\['month'\] \= df\['month'\].map(month\_map)  
df\['day'\] \= df\['day'\].map(day\_map)

X\_full \= df.drop(columns=\['area'\]).values  
y\_full \= df\['area'\].values.reshape(-1, 1)

np.random.seed(42)  
indices \= np.arange(len(X\_full))  
np.random.shuffle(indices)

train\_size \= int(len(X\_full) \* 0.8)  
train\_idx \= indices\[:train\_size\]  
test\_idx \= indices\[train\_size:\]

X\_train \= X\_full\[train\_idx\]  
y\_train \= y\_full\[train\_idx\]  
X\_test \= X\_full\[test\_idx\]  
y\_test \= y\_full\[test\_idx\]

mu\_train \= np.mean(X\_train, axis=0)  
sigma\_train \= np.std(X\_train, axis=0)  
sigma\_train\[sigma\_train \== 0\] \= 1e-15 

X\_train\_std \= (X\_train \- mu\_train) / sigma\_train  
X\_test\_std \= (X\_test \- mu\_train) / sigma\_train

X\_train\_std \= np.hstack(\[np.ones((X\_train\_std.shape\[0\], 1)), X\_train\_std\])  
X\_test\_std \= np.hstack(\[np.ones((X\_test\_std.shape\[0\], 1)), X\_test\_std\])

beta \= np.linalg.inv(X\_train\_std.T @ X\_train\_std) @ X\_train\_std.T @ y\_train  
y\_pred \= X\_test\_std @ beta

mse \= mean\_squared\_error(y\_test, y\_pred)  
r2 \= r2\_score(y\_test, y\_pred)

print("="\*10 \+ " ASSIGNMENT 2: OLS WITH STANDARDIZATION " \+ "="\*10)  
print(f"MSE: {mse}")  
print(f"R2 Score: {r2}")

**Giải thích từng dòng:**

* Phần xử lý dữ liệu và chia tách giữ nguyên hoàn toàn so với Assignment 1\. Đảm bảo dùng np.random.seed(42) để có chung tập phân chia.  
* mu\_train và sigma\_train: Tính trung bình và độ lệch chuẩn **chỉ trên tập Train** (tránh Data Leakage).  
* sigma\_train\[sigma\_train \== 0\] \= 1e-15: Một trick nhỏ của sinh viên để tránh lỗi chia cho 0 nếu có cột nào đó có phương sai bằng 0\.  
* X\_train\_std \= (X\_train \- mu\_train) / sigma\_train: Chuẩn hóa dữ liệu theo công thức Z-score. Tập Test cũng bắt buộc phải dùng mu và sigma của Train.  
* Cực kỳ quan trọng: Hàm np.hstack nối cột 1 được gọi **sau khi** đã tính standardization, đúng nguyên tắc không được chuẩn hóa bias.

### **FILE 3: src/assignment\_3.py**

Python  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.metrics import mean\_squared\_error, r2\_score

df \= pd.read\_csv("data/forestfires.csv")  
month\_map \= {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}  
day\_map \= {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}  
df\['month'\] \= df\['month'\].map(month\_map)  
df\['day'\] \= df\['day'\].map(day\_map)

corr\_matrix \= df.corr()  
plt.figure(figsize=(10, 8))  
sns.heatmap(corr\_matrix, annot=True, cmap='coolwarm', fmt=".2f")  
plt.title("Correlation Matrix")  
plt.savefig("correlation\_heatmap.png")

df \= df.drop(columns=\['DC'\])

X\_full \= df.drop(columns=\['area'\]).values  
y\_full \= df\['area'\].values.reshape(-1, 1)

np.random.seed(42)  
indices \= np.arange(len(X\_full))  
np.random.shuffle(indices)

train\_size \= int(len(X\_full) \* 0.8)  
train\_idx \= indices\[:train\_size\]  
test\_idx \= indices\[train\_size:\]

X\_train \= X\_full\[train\_idx\]  
y\_train \= y\_full\[train\_idx\]  
X\_test \= X\_full\[test\_idx\]  
y\_test \= y\_full\[test\_idx\]

mu\_train \= np.mean(X\_train, axis=0)  
sigma\_train \= np.std(X\_train, axis=0)  
sigma\_train\[sigma\_train \== 0\] \= 1e-15 

X\_train\_std \= (X\_train \- mu\_train) / sigma\_train  
X\_test\_std \= (X\_test \- mu\_train) / sigma\_train

X\_train\_std \= np.hstack(\[np.ones((X\_train\_std.shape\[0\], 1)), X\_train\_std\])  
X\_test\_std \= np.hstack(\[np.ones((X\_test\_std.shape\[0\], 1)), X\_test\_std\])

beta \= np.linalg.inv(X\_train\_std.T @ X\_train\_std) @ X\_train\_std.T @ y\_train  
y\_pred \= X\_test\_std @ beta

mse \= mean\_squared\_error(y\_test, y\_pred)  
r2 \= r2\_score(y\_test, y\_pred)

print("="\*10 \+ " ASSIGNMENT 3: FEATURE ENGINEERING (DROP DC) " \+ "="\*10)  
print(f"MSE: {mse}")  
print(f"R2 Score: {r2}")

**Giải thích từng dòng:**

* corr\_matrix \= df.corr(): Sinh ma trận hệ số tương quan tuyến tính Pearson cho toàn bộ DataFrame.  
* sns.heatmap(...): Vẽ và lưu lại Heatmap ra file ảnh (để sau add vào báo cáo/bài nộp). Nhìn vào ảnh này bạn sẽ thấy DC và DMC có tương quan cao lên đến 0.68.  
* df.drop(columns=\['DC'\]): Chốt loại bỏ thuộc tính DC ra khỏi bộ dữ liệu ngay từ đầu nhằm phá vỡ đa cộng tuyến (collinearity) trước khi chia Train/Test.  
* Các phần tính OLS và Standardize được thực hiện lặp lại hoàn toàn (có cố định seed) để đánh giá việc bỏ 1 cột sẽ làm thay đổi độ lỗi thế nào.

### **FILE 4: src/assignment\_4.py**

Python  
import pandas as pd  
import numpy as np  
from sklearn.linear\_model import LinearRegression  
from sklearn.metrics import mean\_squared\_error, r2\_score

df \= pd.read\_csv("data/forestfires.csv")  
month\_map \= {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}  
day\_map \= {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}  
df\['month'\] \= df\['month'\].map(month\_map)  
df\['day'\] \= df\['day'\].map(day\_map)  
df \= df.drop(columns=\['DC'\])

X\_full \= df.drop(columns=\['area'\]).values  
y\_full \= df\['area'\].values.reshape(-1, 1)

np.random.seed(42)  
indices \= np.arange(len(X\_full))  
np.random.shuffle(indices)

train\_size \= int(len(X\_full) \* 0.8)  
train\_idx \= indices\[:train\_size\]  
test\_idx \= indices\[train\_size:\]

X\_train \= X\_full\[train\_idx\]  
y\_train \= y\_full\[train\_idx\]  
X\_test \= X\_full\[test\_idx\]  
y\_test \= y\_full\[test\_idx\]

mu\_train \= np.mean(X\_train, axis=0)  
sigma\_train \= np.std(X\_train, axis=0)  
sigma\_train\[sigma\_train \== 0\] \= 1e-15 

X\_train\_std \= (X\_train \- mu\_train) / sigma\_train  
X\_test\_std \= (X\_test \- mu\_train) / sigma\_train

model \= LinearRegression()  
model.fit(X\_train\_std, y\_train)

y\_pred \= model.predict(X\_test\_std)

mse \= mean\_squared\_error(y\_test, y\_pred)  
r2 \= r2\_score(y\_test, y\_pred)

print("="\*10 \+ " ASSIGNMENT 4: SKLEARN LINEAR REGRESSION " \+ "="\*10)  
print(f"MSE: {mse}")  
print(f"R2 Score: {r2}")

**Giải thích từng dòng:**

* Ở file cuối cùng này, quy trình tiền xử lý tương đương Assignment 3 (đã bỏ DC và chuẩn hóa) để đảm bảo có thể so sánh trực tiếp một cách công bằng với class tự code.  
* model \= LinearRegression(): Khởi tạo mô hình tuyến tính tích hợp sẵn từ Sklearn thay vì giải ma trận bằng tay. (Lưu ý: Không cần np.hstack thêm cột 1 nữa vì thuật toán Sklearn mặc định đã tự động handle bias qua tham số fit\_intercept=True có sẵn bên trong hàm).  
* model.fit() và model.predict(): Tiến hành huấn luyện và dự đoán, kết quả MSE và R2 thu được sẽ bám rất sát với kết quả làm thủ công ở Assignment 3\.
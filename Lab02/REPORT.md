# Báo cáo Lab02: Phân loại món ăn Việt Nam bằng Deep Learning

---

## 1. Giới thiệu

Trong bài thực hành này, tôi xây dựng và đánh giá các mô hình mạng nơ-ron tích
chập để phân loại hình ảnh món ăn Việt Nam. Lab02 gồm hai bài tập: **LeNet-5**
và **GoogleNet** với các khối Inception.

**Mục tiêu chính:**

- Làm quen với quy trình huấn luyện mô hình phân loại ảnh bằng PyTorch.
- Tìm hiểu cách sử dụng các kiến trúc CNN khác nhau cho bài toán nhận diện
  món ăn.
- Đánh giá mô hình bằng các chỉ số F1-score, precision, recall và accuracy.

---

## 2. Cấu trúc thư mục

```text
Lab02/
├─ dataset/
│  ├─ train/                       # Ảnh huấn luyện, chia theo thư mục lớp
│  └─ test/                        # Ảnh kiểm tra, chia theo thư mục lớp
├─ Assignment01/
│  ├─ LeNet.py                     # Kiến trúc LeNet-5
│  ├─ assignment_01.py             # Luồng chạy Assignment 01
│  └─ engine.py                   # Huấn luyện và đánh giá
├─ Assignment02/
│  ├─ GoogleNet.py                 # GoogleNet với các khối Inception
│  ├─ assignment02.py              # Luồng chạy Assignment 02
│  └─ engine.py                   # Huấn luyện và đánh giá
├─ data_preprocessing.py           # Nạp và tiền xử lý ảnh
└─ REPORT.md                       # Báo cáo thực hành
```

Chạy các assignment từ thư mục gốc của repository bằng PowerShell:

```powershell
python Lab02\Assignment01\assignment_01.py --train-dir Lab02\dataset\train --test-dir Lab02\dataset\test --epochs 10
python Lab02\Assignment02\assignment02.py --train-dir Lab02\dataset\train --test-dir Lab02\dataset\test
```

---

## 3. Mô hình và quy trình thực hiện

### 3.1. GoogleNet

Assignment 02 sử dụng mô hình GoogleNet được xây dựng trong
`Assignment02/GoogleNet.py`. Mô hình gồm các lớp tích chập, các khối Inception
và lớp phân loại cuối cùng cho **21 lớp**.

Mỗi khối Inception xử lý đầu vào qua bốn nhánh song song: tích chập 1 × 1,
nhánh tích chập 1 × 1 kết hợp 3 × 3, nhánh tích chập 1 × 1 kết hợp 5 × 5, và
nhánh max pooling kết hợp tích chập 1 × 1. Đầu ra các nhánh được nối theo
chiều kênh. Sau các khối trích xuất đặc trưng, mô hình dùng global average
pooling, dropout và các lớp fully connected để dự đoán nhãn.

### 3.2. Huấn luyện

- Cả hai mô hình dùng `CrossEntropyLoss`, Adam, batch size 64 và được đánh giá
  trên tập validation sau mỗi epoch.
- Các lần chạy được báo cáo sử dụng 10 epoch. Thiết bị được chọn tự động:
  CUDA nếu khả dụng, nếu không thì CPU.
- Assignment 02 lưu checkpoint có validation macro F1 tốt nhất tại
  `checkpoint/best_model.pth` và nạp checkpoint này trước khi đánh giá test.
- Assignment 01 cũng lưu checkpoint validation tốt nhất, nhưng lần chạy được
  báo cáo đánh giá test bằng trọng số sau epoch cuối, không nạp lại checkpoint.

---

## 4. Tiền xử lý dữ liệu

Dữ liệu VinaFood21 được nạp bằng `torchvision.datasets.ImageFolder`; mỗi thư
mục con tương ứng với một lớp. Các ảnh được:

1. Thay đổi kích thước thành **64 × 64**.
2. Chuyển thành tensor.
3. Chuẩn hóa từng kênh màu với mean và standard deviation bằng 0.5.

Tập train được chia ngẫu nhiên thành tập huấn luyện (80%) và validation (20%),
sử dụng seed 42. Tập test được giữ riêng để đánh giá cuối cùng.

---

## 5. Kết quả Assignment 02

### 5.1. Kết quả theo epoch

| Epoch | Validation F1 (macro) | Validation Accuracy |
| ----: | --------------------: | ------------------: |
| 1 | 0.0515 | 0.1170 |
| 2 | 0.0808 | 0.1966 |
| 3 | 0.1198 | 0.2210 |
| 4 | 0.1363 | 0.2210 |
| 5 | 0.1513 | 0.2409 |
| 6 | 0.1631 | 0.2409 |
| 7 | 0.1439 | 0.2115 |
| 8 | 0.2111 | 0.2832 |
| 9 | **0.2291** | **0.2937** |
| 10 | 0.2108 | 0.2912 |

Validation macro F1 cao nhất đạt **0.2291 tại epoch 9**. Vì vậy checkpoint
được chọn để đánh giá test là checkpoint của epoch này.

### 5.2. Đánh giá trên tập test

| Chỉ số | Kết quả |
| :--- | ---: |
| **F1-score (Macro)** | 0.2142 |
| **Precision (Macro)** | 0.2324 |
| **Recall (Macro)** | 0.2366 |
| **Accuracy** | 0.2769 |

### 5.3. Nhận xét

- Validation macro F1 tăng nhìn chung trong quá trình huấn luyện, từ **0.0515**
  ở epoch 1 lên mức cao nhất **0.2291** ở epoch 9. Kết quả giảm nhẹ ở epoch
  cuối nên việc chọn checkpoint theo validation F1 giúp giữ lại mô hình tốt
  hơn trong lần chạy này.
- Accuracy trên test đạt **0.2769**; macro F1 đạt **0.2142**. Đây là kết quả
  ban đầu còn hạn chế đối với bài toán phân loại 21 lớp, cho thấy mô hình vẫn
  gặp khó khăn khi phân biệt các lớp món ăn.
- Log huấn luyện có cảnh báo từ PIL về ảnh palette có transparency. Cảnh báo
  xuất hiện trong lúc tải ảnh nhưng không làm dừng quá trình huấn luyện.
- Kết quả có thể được cải thiện thêm bằng cách huấn luyện lâu hơn, điều chỉnh
  siêu tham số, tăng cường dữ liệu hoặc sử dụng mô hình tiền huấn luyện; cần
  kiểm chứng các thay đổi này bằng thực nghiệm.

---

## 6. Assignment 01 - LeNet-5

Assignment 01 sử dụng LeNet-5 để phân loại cùng bộ dữ liệu. Mô hình được huấn
luyện trong 10 epoch. Bảng dùng cách đánh số epoch bắt đầu từ 1 để thống nhất
với Assignment 02; log gốc đánh số từ 0 và không có dòng cho epoch thứ 7.

| Epoch | Validation F1 (macro) | Validation Precision (macro) | Validation Recall (macro) | Validation Accuracy |
| ----: | --------------------: | --------------------------: | ------------------------: | ------------------: |
| 1 | 0.0784 | 0.0875 | 0.1087 | 0.1563 |
| 2 | 0.1252 | 0.1598 | 0.1510 | 0.1996 |
| 3 | 0.1554 | 0.2726 | 0.1835 | 0.2270 |
| 4 | 0.1828 | 0.2455 | 0.2022 | 0.2474 |
| 5 | 0.2271 | 0.2498 | 0.2403 | 0.2693 |
| 6 | 0.2454 | 0.2670 | 0.2626 | 0.2763 |
| 8 | 0.2563 | 0.2883 | 0.2666 | 0.2847 |
| 9 | **0.2667** | 0.2869 | 0.2634 | 0.2837 |
| 10 | 0.2599 | 0.2884 | 0.2627 | 0.2787 |

Validation macro F1 cao nhất đạt **0.2667 tại epoch 9**. Tuy nhiên, lần chạy
này đánh giá trên tập test bằng trọng số sau epoch cuối (epoch 10), không nạp
lại checkpoint tốt nhất.

### 6.1. Đánh giá trên tập test

Hàm đánh giá của Assignment 01 trả về các chỉ số theo thứ tự F1, accuracy,
precision, recall. Các giá trị test dưới đây đã được sắp xếp lại theo đúng
nhãn; precision và accuracy đều làm tròn thành 0.2481:

| Chỉ số | Kết quả |
| :--- | ---: |
| **F1-score (Macro)** | 0.2248 |
| **Precision (Macro)** | 0.2481 |
| **Recall (Macro)** | 0.2331 |
| **Accuracy** | 0.2481 |

### 6.2. Nhận xét

- Validation macro F1 tăng từ **0.0784** ở epoch 1 lên mức cao nhất **0.2667**
  ở epoch 9, sau đó giảm nhẹ ở epoch 10.
- Kết quả test có macro F1 **0.2248** và accuracy **0.2481**, cho thấy mô hình
  còn gặp khó khăn khi phân loại 21 lớp.
- Cảnh báo sklearn cho biết một số lớp không có mẫu nào được dự đoán; precision
  của các lớp đó được tính bằng 0 trong phép tính macro. Cảnh báo PIL về ảnh
  palette có transparency cũng xuất hiện khi tải ảnh, nhưng không làm dừng
  quá trình huấn luyện.
- Kết quả test ở đây thuộc trọng số epoch 10. Có thể đánh giá lại checkpoint
  của epoch 9 để so sánh trực tiếp mô hình có validation F1 cao nhất.

---

## 7. Kết luận

Hai bài tập đã hoàn thiện quy trình nạp và tiền xử lý ảnh, huấn luyện mô hình
và đánh giá trên tập test. Trong các lần chạy được ghi nhận, GoogleNet đạt
test accuracy **0.2769** và macro F1 **0.2142**; LeNet-5 đạt test accuracy
**0.2481** và macro F1 **0.2248**. Đây là các mốc kết quả ban đầu, chưa phải
so sánh hoàn toàn tương đương: GoogleNet được đánh giá bằng checkpoint tốt
nhất (epoch 9), còn LeNet-5 được đánh giá bằng trọng số epoch cuối (epoch 10).

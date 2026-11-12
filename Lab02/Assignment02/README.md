# Assignment 02 GoogLeNet rút gọn

Theo B.4 và C.1 Bài 2 trong Lab 2, mô hình dùng **3 Inception block**,
ảnh RGB 64×64 và 21 lớp. Mỗi block có bốn nhánh Conv 1×1,
1×1 → 3×3, 1×1 → 5×5, MaxPool → 1×1; nối bằng `torch.cat(..., dim=1)`.
Conv dùng bias và ReLU theo B.4. Không dùng auxiliary classifier.
Tổng số tham số của cấu hình 21 lớp này là **100.297**.

| Stage | Output C×H×W | Cách tính channel |
| --- | --- | --- |
| Stem | 64×16×16 | Conv 3×3 → pool → Conv 3×3 → pool |
| Inception 1 | 56×16×16 | 16 + 24 + 8 + 8 = 56 |
| Inception 2 | 112×16×16 | 32 + 48 + 16 + 16 = 112 |
| MaxPool | 112×8×8 | Giữ số channel |
| Inception 3 | 192×8×8 | 64 + 64 + 32 + 32 = 192 |
| Global average pooling | 192×1×1 | `AdaptiveAvgPool2d(1)` |
| Classifier | 21 | Linear(192, 21) |

Chỉ chuyển vector sau global average pooling sang dạng `[N, 192]`,
không flatten toàn bộ feature map không gian vào lớp Linear.
Code in shape, số tham số từng stage và đối chiếu channel tính tay với output thật.

Chạy từ repository root:

```powershell
python Lab02/Assignment02/assignment02.py --epochs 10
```

Mặc định: Adam lr 0.001, batch size 64, seed 42; dùng chung
`data_preprocessing.processing` với Assignment01 hiện tại (split 80/20,
resize 64×64, normalize mean/std 0.5). Các tham số `--train-dir`, `--test-dir`,
`--seed`, `--batch-size`, `--lr`, `--output-dir` cho phép thay cấu hình.
Checkpoint có validation macro-F1 cao nhất được nạp trước khi đánh giá test.

Trong `results/` có `best_model.pth`, `metrics.json`, `history.csv`,
`layers.csv`, `inception_channels.csv`, `per_class.csv`,
`confusion_matrix.csv`, `confusion_matrix.png` và `loss.png`.
Ma trận có 21 hàng nhãn thật và 21 cột nhãn dự đoán, theo thứ tự
`ImageFolder.classes`. Precision, recall, F1 đều lấy macro trên đủ 21 lớp.

`comparison.csv` và `comparison.md` so sánh số tham số, giây/epoch,
accuracy và macro-F1 với hai bản LeNet. Code đọc `Assignment01/results/avg/metrics.json`
và `Assignment01/results/max/metrics.json`; dùng `--lenet-results-dir` để đổi vị trí.
Nếu chưa có kết quả LeNet, bảng ghi `Pending run`, không điền số giả.
Thời gian mỗi epoch gồm train và validation, chưa gồm lưu checkpoint;
cần dùng cùng máy, dữ liệu, preprocessing và cấu hình để so sánh.

Mô hình mới có kiến trúc khác bản 9 block trước đây nên cần huấn luyện lại.
Kết quả GoogLeNet cũ trong `../REPORT.md` không đại diện cho cấu hình mới.
Đối chiếu hai ví dụ B.4 bằng `Inception(1,16,16,24,4,8,8)` và
`Inception(56,32,32,48,8,16,16)`: lần lượt 4.376 và 22.104 tham số.

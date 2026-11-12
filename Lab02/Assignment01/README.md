# Assignment 01 LeNet-5 trên VinaFood21

Triển khai theo B.3 và C.1 Bài 1 trong đề Lab 2: ảnh RGB 64×64,
21 lớp, ReLU, Adam, so sánh AvgPool2d và MaxPool2d. Cả hai cấu hình có
339.421 tham số, flatten 2.704 chiều. F5 chiếm 324.600 tham số.

Chạy từ thư mục gốc repository:

```powershell
python Lab02/Assignment01/assignment_01.py --epochs 20
```

Mặc định chạy lần lượt cả hai bản pooling, batch size 32, Adam lr 0.001.
Có thể dùng `--pool avg` hoặc `--pool max` để chạy riêng; `--epochs 1`
chỉ phù hợp để kiểm tra luồng chạy, chưa đủ để kết luận chất lượng mô hình.
Đường dẫn dữ liệu mặc định được tính theo vị trí script, nên có thể chạy từ
thư mục khác. Dùng `--train-dir`, `--test-dir`, `--output-dir` để đổi đường dẫn.

Hai lần chạy dùng seed 42, cùng phép chia train/validation 80/20,
cùng thứ tự batch và trọng số khởi tạo. Theo A.2, chỉ train được augmentation
(crop, lật ngang, xoay và color jitter); validation/test chỉ resize và chuẩn hóa.
Cả ba dùng mean/std ImageNet. Mỗi lần đánh giá test đều nạp checkpoint có
validation macro-F1 tốt nhất. Macro precision, recall, F1 tính trên đủ 21 lớp,
với điểm 0 cho trường hợp không xác định.

Kết quả được sinh sau khi chạy tại `results/avg/` và `results/max/`:

- `best_model.pth`: trọng số và trạng thái optimizer của epoch tốt nhất.
- `metrics.json`: bốn độ đo test, tên lớp, cấu hình và classification report.
- `per_class.csv`: precision, recall, F1, support theo từng lớp.
- `confusion_matrix.csv` và `confusion_matrix.png`: ma trận 21×21, hàng là
  nhãn thật, cột là nhãn dự đoán, theo thứ tự `ImageFolder.classes`.
- `layers.csv`: shape và số tham số từng layer.
- `history.csv` và `loss.png`: loss train/validation, validation F1 và thời gian
  mỗi epoch (train + validation, không gồm ghi checkpoint).

`results/comparison.csv` và `results/comparison.md` tổng hợp bốn độ đo,
số tham số, epoch được chọn và thời gian trung bình của các bản pooling
trong lần gọi script đó. Chạy riêng một bản sẽ ghi lại bảng tổng hợp chỉ gồm bản đó.

Các kết quả cũ trong `../REPORT.md` thuộc lần chạy trước, dùng preprocessing
và cách chọn trọng số khác. Chỉ cập nhật bảng báo cáo sau khi có kết quả chạy mới;
không dùng số cũ để kết luận chênh lệch AvgPool/MaxPool.

Thư viện cần có: `torch`, `torchvision`, `scikit-learn`, `matplotlib`.

# DS201 Lab 2 — Bài 3: ResNet-18

Cấu trúc `Assignment03` giống Bài 1: `ResNet.py`, `assignment_03.py`, `data_preprocessing.py`, `engine.py`, `reporting.py`. Các kết quả nằm trong `Assignment03/results/`.

## Dataset

Đặt `dataset/train/<ten_lop>/*.jpg` và `dataset/test/<ten_lop>/*.jpg` trong thư mục `Lab02` (ngang cấp `Assignment03`), hoặc truyền `--train-dir` và `--test-dir`.

## Chạy trên Colab

```python
%cd /content/drive/MyDrive/SEM_05/DS201/Lab02
!python Assignment03/assignment_03.py --train-dir /content/dataset/train --test-dir /content/dataset/test --epochs 10 --mode both
```

Để chuẩn bị `/content/dataset`, chạy `!cp -r dataset /content/dataset` từ thư mục Lab02 (lệnh copy có thể mất thời gian). Không bắt buộc copy; bỏ hai tham số đường dẫn để đọc trực tiếp từ Google Drive.

`--mode` nhận `both` (mặc định), `no_aug`, `aug`. Có thể thay đổi `--batch-size`, `--epochs`, `--img-size` (96 hoặc 112), `--lr`, `--num-workers`, `--output-dir`.

## Kết quả

Mỗi run lưu: best_model.pth, history.csv, loss.png, confusion_matrix.csv/png, per_class.csv, layers.csv, metrics.json. Bảng comparison.csv/md chứa chênh lệch Macro-F1. Trường `loss_divergence_epoch` là heuristic (epoch đầu tiên val loss tăng trong khi train loss giảm so với epoch trước), cần kiểm chứng bằng đồ thị và nhận xét trong báo cáo.

Ảnh validation và test không augmentation, train augment chỉ trong mode `aug`. Cùng index split 80/20 seed 42 cho cả hai run. Không dùng pretrained weights.

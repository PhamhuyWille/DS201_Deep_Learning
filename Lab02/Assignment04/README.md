# DS201 Lab 2 — Bài 4: Transfer Learning ResNet-50

Bám theo phần C.2 (Bài 4) và B.6 trong tài liệu Lab 2: model **`microsoft/resnet-50` từ Hugging Face**; thay classifier bằng `nn.Linear(2048, 21)`; input 224×224; feature extraction (`freeze_backbone=True`, Adam `1e-3`) và full fine-tuning (`freeze_backbone=False`, Adam `1e-4`).

## Cấu trúc

```text
Lab02/
├── dataset/
│   ├── train/<ten_lop>/*.jpg
│   └── test/<ten_lop>/*.jpg
└── Assignment04/
    ├── pretrained_resnet.py
    ├── assignment_04.py
    ├── data_preprocessing.py
    ├── engine.py
    ├── reporting.py
    └── results/                   # tạo khi chạy
```

## Cài đặt và chạy Google Colab

```python
%cd /content/drive/MyDrive/SEM_05/DS201/Lab02
!pip -q install -U transformers torchvision scikit-learn tqdm matplotlib
# Khuyến nghị dữ liệu cục bộ, không đọc hàng nghìn ảnh trực tiếp từ Drive:
!cp -r dataset /content/dataset
!python Assignment04/assignment_04.py --train-dir /content/dataset/train --test-dir /content/dataset/test --epochs 10 --batch-size 32 --mode both
```

Chỉ chạy feature extraction: `--mode feature_extraction`; chỉ chạy fine-tuning: `--mode fine_tuning`. Nếu GPU hết VRAM, giảm batch size xuống 16 hoặc 8. `--num-workers 2` mặc định; dùng `--num-workers 0` nếu Colab gặp vấn đề multiprocessing. Lần đầu chạy cần Internet để tải pretrained weights và image processor từ Hugging Face (model có thể yêu cầu kết nối mạng của Colab).

## Đầu ra

Mỗi thí nghiệm có `best_model.pth`, `history.csv`, `loss.png`, `per_class.csv`, `confusion_matrix.csv/png`, `layers.csv`, `metrics.json`. Trong `results/` có `comparison.csv`, `comparison.md`, `four_models.csv` (đọc các metrics của Bài 1/2/3 nếu tìm thấy đúng đường dẫn, nếu không sẽ để trống).

Tập train gốc tách 80/20 thành train/validation với seed 42; dùng test riêng. Cả hai chế độ dùng cùng cách chia, transforms và 21 nhãn; F1-macro của validation dùng để chọn checkpoint. Feature extraction cố định cả tham số backbone và các thống kê BatchNorm trong backbone.

**Chú ý:** `--mode both` tải lại pretrained weights để khởi tạo độc lập cho mỗi thí nghiệm; checkpoint và optimizer không được tái sử dụng từ chế độ kia. Không có số liệu giả lập. Script chưa được huấn luyện trên dataset thật; kiểm tra cú pháp thực hiện trước khi bàn giao.

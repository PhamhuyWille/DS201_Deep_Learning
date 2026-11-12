# DS201 Lab 02 Phân loại món ăn Việt Nam

Lab 02 so sánh LeNet-5, GoogLeNet rút gọn, ResNet-18 huấn luyện từ đầu và ResNet-50 tiền huấn luyện trên VinaFood21. Trong bảy cấu hình đã chạy, ResNet-50 fine-tuning đạt kết quả test cao nhất với **82,43% accuracy** và **0,8251 macro-F1**.

## Dữ liệu và đánh giá

- VinaFood21 có 21 lớp. Tập train gốc gồm 10.044 ảnh, được chia thành 8.035 ảnh huấn luyện và 2.009 ảnh validation theo tỷ lệ 80/20 với seed 42. Tập test riêng gồm 6.682 ảnh. [Thông tin bộ dữ liệu](dataset/README.txt)
- Các lần chạy trong bảng đều gồm 10 epoch. Mỗi cấu hình được đánh giá trên test bằng checkpoint có macro-F1 cao nhất trên validation. Macro-F1 là trung bình F1 của 21 lớp với trọng số bằng nhau.
- Giây/epoch là thời gian trung bình để huấn luyện và đánh giá validation, đo trước khi ghi checkpoint. Các lần chạy dùng môi trường và batch size khác nhau, nên cột này phản ánh thời gian ghi nhận của từng lần chạy.

## Chạy thí nghiệm

Chạy các lệnh từ thư mục `Lab02`. Dữ liệu mặc định nằm ở `dataset/train/<tên_lớp>/` và `dataset/test/<tên_lớp>/`.

```powershell
python -m pip install torch torchvision numpy scikit-learn matplotlib tqdm transformers

python Assignment01/assignment_01.py --epochs 10 --batch-size 64 --pool avg
python Assignment01/assignment_01.py --epochs 10 --batch-size 64 --pool max
python Assignment02/assignment02.py --epochs 10 --batch-size 64
python Assignment03/assignment_03.py --epochs 10 --batch-size 32 --img-size 96 --mode no_aug
python Assignment03/assignment_03.py --epochs 10 --batch-size 32 --img-size 96 --mode aug
```

Ở bản mã hiện tại, script Bài 4 dừng khi đọc `args.num_workers` vì tham số này chưa được khai báo. Sau khi xử lý lỗi đó, dùng lệnh dưới đây để chạy cả hai chế độ. Lần đầu cần kết nối mạng để tải trọng số và cấu hình tiền xử lý của `microsoft/resnet-50`. [Script Bài 4](Assignment04/assignment_04.py)

```powershell
python Assignment04/assignment_04.py --epochs 10 --batch-size 32 --mode both
```

Mỗi lần chạy lưu kết quả trong thư mục `results/` của bài tương ứng. File `metrics.json` có cấu hình, checkpoint được chọn và các chỉ số test. File `history.csv` có loss, chỉ số validation và thời gian từng epoch. File `per_class.csv` và `confusion_matrix.csv` cho phép xem kết quả theo lớp.

## Nhận xét bảng kết quả

- **LeNet-5 và GoogLeNet.** Với cùng 339.421 tham số, AvgPool đạt macro-F1 cao hơn MaxPool 0,0161. Trong nhóm dùng ảnh 64×64, GoogLeNet có accuracy và macro-F1 cao nhất. So với LeNet-5 AvgPool, GoogLeNet giảm 70,45% số tham số và tăng macro-F1 thêm 0,0270. [LeNet-5 AvgPool](Assignment01/results/avg/metrics.json) · [LeNet-5 MaxPool](Assignment01/results/max/metrics.json) · [GoogLeNet](Assignment02/results/metrics.json)
- **ResNet-18 huấn luyện từ đầu.** Bản có augmentation dẫn đầu nhóm huấn luyện từ đầu với macro-F1 0,3744, cao hơn GoogLeNet 0,1032. Trong cặp ResNet-18, bản có augmentation tăng accuracy test 1,59 điểm phần trăm và macro-F1 test 0,0039. Macro-F1 tốt nhất trên validation giảm từ 0,4292 xuống 0,4019, nên lợi thế quan sát được trong lần chạy này nằm ở tập test. [GoogLeNet](Assignment02/results/metrics.json) · [Không augmentation](Assignment03/results/no_aug/metrics.json) · [Có augmentation](Assignment03/results/aug/metrics.json)
- **ResNet-50 feature extraction.** Chỉ huấn luyện 43.029 tham số của lớp phân loại, cấu hình này đạt macro-F1 0,7534, khoảng gấp đôi ResNet-18 có augmentation. Hai cấu hình khác nhau về kiến trúc, kích thước ảnh đầu vào và cách khởi tạo trọng số. [ResNet-18 có augmentation](Assignment03/results/aug/metrics.json) · [Feature extraction](Assignment04/results/feature_extraction/metrics.json)
- **ResNet-50 fine-tuning.** So với feature extraction, cập nhật toàn bộ mô hình tăng accuracy thêm 7,71 điểm phần trăm và macro-F1 thêm 0,0717. Thời gian trung bình tăng từ 202,84 lên 270,18 giây/epoch, tương đương thêm 67,34 giây mỗi epoch trong hai lần chạy ghi nhận. [Feature extraction](Assignment04/results/feature_extraction/metrics.json) · [Fine-tuning](Assignment04/results/fine_tuning/metrics.json)

## Bảng tổng hợp kết quả

Accuracy và macro-F1 được đo trên tập test. Cột Tham số ghi tổng số tham số của mô hình.

| Mô hình | Resize | Tham số | Giây/epoch | Accuracy | macro-F1 |
| --- | :---: | ---: | ---: | ---: | ---: |
| LeNet-5 điều chỉnh (AvgPool) | 64×64 | 339.421 | 156,92 | 26,20% | 0,2442 |
| LeNet-5 điều chỉnh (MaxPool) | 64×64 | 339.421 | 164,84 | 24,05% | 0,2280 |
| GoogLeNet rút gọn | 64×64 | 100.297 | 139,35 | 28,87% | 0,2712 |
| ResNet-18 từ đầu (không augmentation) | 96×96 | 11.187.285 | 129,72 | 39,09% | 0,3704 |
| ResNet-18 từ đầu (có augmentation) | 96×96 | 11.187.285 | 140,13 | 40,68% | 0,3744 |
| ResNet-50 pretrained (feature extraction) | 224×224 | 23.551.061 | 202,84 | 74,72% | 0,7534 |
| ResNet-50 pretrained (fine-tuning) | 224×224 | 23.551.061 | 270,18 | **82,43%** | **0,8251** |

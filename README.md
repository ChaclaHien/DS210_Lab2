# Lab 2: Mạng neural tích chập (CNN)

Bài thực hành 2 – xây dựng, huấn luyện và đánh giá các kiến trúc CNN kinh điển bằng PyTorch. Tất cả mô hình dùng **Adam** làm optimizer và được đánh giá bằng **accuracy, precision, recall, F1 (macro)**.

| Bài | Mô hình | Bộ dữ liệu |
|-----|---------|-----------|
| 1 | LeNet | MNIST |
| 2 | GoogLeNet | VinaFood21 |
| 3 | ResNet-18 | VinaFood21 |
| 4 | ResNet-50 pretrained (HuggingFace) fine-tune | VinaFood21 |

## Cấu trúc thư mục

```
.
├── assignment-2.py          # File chạy chính (Bài 1 → 4)
├── data_utils/
│   ├── __init__.py
│   ├── mnist_dataset.py     # DataLoader cho MNIST (Bài 1)
│   └── vinafood_dataset.py  # Transform, DataLoader cho VinaFood21 (Bài 2, 3, 4)
├── models/
│   ├── __init__.py
│   ├── lenet.py             # LeNet
│   ├── googlenet.py         # InceptionBlock + GoogLeNet
│   ├── resnet18.py          # ResidualBlock + ResNet18
│   └── resnet50.py          # Tải & fine-tune ResNet50 từ HuggingFace
├── utils/
│   ├── __init__.py
│   └── engine.py            # Hàm train() và evaluate() dùng chung
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu Python 3.8+. Nên dùng GPU (CUDA) để huấn luyện Bài 2, 3, 4; code tự động chọn `cuda` nếu có, ngược lại dùng `cpu`.

## Chuẩn bị dữ liệu

- **MNIST (Bài 1):** tự động tải về thư mục `data/` ở lần chạy đầu tiên.
- **VinaFood21 (Bài 2, 3, 4):** tải tại [Google Drive](https://drive.google.com/file/d/1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR/view?usp=share_link), giải nén thành cấu trúc:

```
VinaFood21/
├── train/
│   ├── <class_1>/*.jpg
│   └── ...
└── test/
    ├── <class_1>/*.jpg
    └── ...
```

## Cách chạy

```bash
# Chạy từng bài
python assignment-2.py --task 1
python assignment-2.py --task 2 --data_root /duong/dan/toi/VinaFood21
python assignment-2.py --task 3 --data_root /duong/dan/toi/VinaFood21
python assignment-2.py --task 4 --data_root /duong/dan/toi/VinaFood21

# Chạy tất cả
python assignment-2.py --task all --data_root /duong/dan/toi/VinaFood21
```

Tham số:

- `--task {1,2,3,4,all}`: bài cần chạy (mặc định `all`).
- `--data_root`: đường dẫn đến thư mục `VinaFood21` (mặc định là đường dẫn trên Kaggle).

Bài 4 cần Internet ở lần chạy đầu để tải trọng số `microsoft/resnet-50` từ HuggingFace.

## Cấu hình huấn luyện

| Bài | Batch size | Learning rate | Epochs | Ghi chú |
|-----|-----------|---------------|--------|---------|
| 1 | 256 | 1e-3 | 10 | Chia train/val = 50000/10000, đánh giá trên tập test |
| 2 | 64 | 1e-4 | 10 | Conv đầu `padding=3`, MaxPool `ceil_mode=True` |
| 3 | 64 | 1e-3 | 10 | MaxPool (k=3, s=2, p=0) giữa các nhóm residual block |
| 4 | 32 | 1e-4 | 10 | Fine-tune toàn bộ mạng, classifier mới 21 lớp |

Với VinaFood21, tập test được dùng làm tập đánh giá sau mỗi epoch.

## Kết quả (trên tập test)

| Mô hình | Accuracy | Precision | Recall | F1 (macro) |
|---------|----------|-----------|--------|-----------|
| LeNet (MNIST) | 0.9660 | 0.9661 | 0.9658 | 0.9658 |
| GoogLeNet (VinaFood21) | 0.2406 | 0.1994 | 0.2050 | 0.1901 |
| ResNet-18 (VinaFood21) | 0.5013 | 0.5600 | 0.5249 | 0.4984 |
| ResNet-50 fine-tune (VinaFood21) | 0.8497 | 0.8492 | 0.8500 | 0.8490 |

> Kết quả có thể thay đổi nhẹ giữa các lần chạy do khởi tạo trọng số và chia dữ liệu ngẫu nhiên.

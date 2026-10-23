"""Lab 2: Mạng neural tích chập (LeNet, GoogLeNet, ResNet-18, ResNet-50 fine-tune).

Ví dụ:
    python assignment-2.py --task 1
    python assignment-2.py --task 2 --data_root /path/to/VinaFood21
    python assignment-2.py --task all --data_root /path/to/VinaFood21
"""
import argparse

from utils.engine import device, train, evaluate
from data_utils.mnist_dataset import get_mnist_loaders
from data_utils.vinafood_dataset import get_vinafood_loaders, get_hf_loaders
from models.lenet import LeNet
from models.googlenet import GoogLeNet
from models.resnet18 import ResNet18
from models.resnet50 import build_resnet50, finetune_resnet50


def bai1():
    """Bài 1: LeNet trên MNIST."""
    train_loader, val_loader, test_loader = get_mnist_loaders(root="data", batch_size=256)
    lenet = LeNet().to(device)
    train(lenet, train_loader, val_loader, epochs=10)
    acc, prec, rec, f1 = evaluate(lenet, test_loader)
    print(f"\n[LeNet Test] acc={acc:.4f} | precision={prec:.4f} | recall={rec:.4f} | f1-macro={f1:.4f}")


def bai2(data_root):
    """Bài 2: GoogLeNet trên VinaFood21."""
    food_train_loader, food_test_loader, num_classes = get_vinafood_loaders(data_root)
    googlenet = GoogLeNet(num_classes=num_classes).to(device)
    train(googlenet, food_train_loader, food_test_loader, epochs=10, lr=1e-4)
    acc, prec, rec, f1 = evaluate(googlenet, food_test_loader)
    print(f"\n[GoogLeNet Test] acc={acc:.4f} | precision={prec:.4f} | recall={rec:.4f} | f1-macro={f1:.4f}")


def bai3(data_root):
    """Bài 3: ResNet-18 trên VinaFood21."""
    food_train_loader, food_test_loader, num_classes = get_vinafood_loaders(data_root)
    resnet18 = ResNet18(num_classes=num_classes).to(device)
    train(resnet18, food_train_loader, food_test_loader, epochs=10)
    acc, prec, rec, f1 = evaluate(resnet18, food_test_loader)
    print(f"\n[ResNet-18 Test] acc={acc:.4f} | precision={prec:.4f} | recall={rec:.4f} | f1={f1:.4f}")


def bai4(data_root):
    """Bài 4: Fine-tune ResNet50 pretrained (HuggingFace) trên VinaFood21."""
    _, _, num_classes = get_vinafood_loaders(data_root)  # lấy NUM_CLASSES như notebook
    processor, resnet50 = build_resnet50(num_classes, device)
    hf_train_loader, hf_test_loader = get_hf_loaders(data_root, processor)
    finetune_resnet50(resnet50, hf_train_loader, hf_test_loader, device, epochs=10, lr=1e-4)


def main():
    parser = argparse.ArgumentParser(description="Lab 2 - CNN")
    parser.add_argument("--task", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Bài cần chạy (1: LeNet, 2: GoogLeNet, 3: ResNet-18, 4: ResNet50 fine-tune)")
    parser.add_argument("--data_root", type=str,
                        default="/kaggle/input/datasets/phmnguynhin/vinafood21/VinaFood21",
                        help="Thư mục VinaFood21 (chứa train/ và test/)")
    args = parser.parse_args()

    print(f"Device: {device}")
    if args.task in ("1", "all"):
        bai1()
    if args.task in ("2", "all"):
        bai2(args.data_root)
    if args.task in ("3", "all"):
        bai3(args.data_root)
    if args.task in ("4", "all"):
        bai4(args.data_root)


if __name__ == "__main__":
    main()

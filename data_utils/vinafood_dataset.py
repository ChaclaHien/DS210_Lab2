import os

from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def rgb_loader(path):
    return PILImage.open(path).convert("RGB")


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_vinafood_loaders(root):
    """Loader cho Bài 2, 3 (GoogLeNet, ResNet-18). root chứa 2 thư mục train/ và test/."""
    food_train = ImageFolder(os.path.join(root, "train"), transform=train_transform, loader=rgb_loader)
    food_test  = ImageFolder(os.path.join(root, "test"),  transform=test_transform,  loader=rgb_loader)
    food_train_loader = DataLoader(food_train, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
    food_test_loader  = DataLoader(food_test,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    num_classes = len(food_train.classes)  # 21
    return food_train_loader, food_test_loader, num_classes


class VinaFoodDataset(Dataset):
    """Dataset dùng AutoImageProcessor của HuggingFace (Bài 4)."""
    def __init__(self, root, processor):
        self.processor = processor
        self.samples   = []
        classes = sorted(os.listdir(root))
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), i))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img    = PILImage.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), label


def get_hf_loaders(root, processor):
    hf_train = VinaFoodDataset(os.path.join(root, "train"), processor)
    hf_test  = VinaFoodDataset(os.path.join(root, "test"),  processor)
    hf_train_loader = DataLoader(hf_train, batch_size=32, shuffle=True,  num_workers=2)
    hf_test_loader  = DataLoader(hf_test,  batch_size=32, shuffle=False, num_workers=2)
    return hf_train_loader, hf_test_loader

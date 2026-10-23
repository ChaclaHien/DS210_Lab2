from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(root="data", batch_size=256):
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=root, train=True,  download=True, transform=transform)
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_set, val_set = random_split(train_data, [50000, 10000])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)
    test_loader  = DataLoader(test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader

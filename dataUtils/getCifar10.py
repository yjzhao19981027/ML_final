from torchvision import datasets
from torchvision import transforms


def getCifar10(train=True):
    #   data of source domain
    img_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='data/',
                                     train=train,
                                     transform=img_transform,
                                     download=True)
    return train_dataset

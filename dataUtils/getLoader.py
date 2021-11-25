from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def getLoader(img_size=32, batch_size=64, train=True):
    #   data of source domain
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    mnist_dataset = datasets.MNIST(
        root='./data',
        train=train,
        transform=img_transform,
        download=True
    )

    mnist_loader = DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    return mnist_loader

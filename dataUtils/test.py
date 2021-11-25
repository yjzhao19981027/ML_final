import torch
import torchvision
import torchvision.transforms as transforms

# 下载CIFAR-10数据集到当前data文件夹中
train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
patch1 = []
patch2 = []
# 从本地硬盘上读取一条数据 (包括1张图像及其对应的标签)
for i in range(len(train_dataset)):
    image, label = train_dataset[i]
    new_data = image[:, :20, :], label
    patch1.append(new_data)
    new_data = image[:, :, 12:], label
    patch2.append(new_data)
print(len(patch1))
train_loader = torch.utils.data.DataLoader(dataset=patch2,
                                           batch_size=64, #该参数表示每次读取的批样本个数
                                           shuffle=True)  #该参数表示读取时是否打乱样本顺序

# 创建迭代器
data_iter = iter(train_loader)

# 当迭代开始时, 队列和线程开始读取数据
images, labels = data_iter.next()

print(images.size())  # 输出 torch.Size([64, 3, 32, 32])
print(labels.size())  # 输出 torch.Size([64])
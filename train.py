import os
import sys
import torch
import torch.nn as nn
from test import test
import torch.optim as optim
from LeNet import LeNet
from getLoader import getLoader

# initialize para
logs_path = 'logs'
cuda = True if torch.cuda.is_available() else False
lr = 1e-3
batch_size = 128
image_size = 32
n_epoch = 20
weight_decay = 1e-5

# load model
net = LeNet()

# load dataLoader
dataLoader = getLoader(image_size, batch_size, train=True)

# initialize optimizer
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

loss = nn.CrossEntropyLoss()

if cuda:
    net = net.cuda()
    loss = loss.cuda()

best_acc = 0.0
for epoch in range(n_epoch):
    len_dataloader = len(dataLoader)
    data_iter = iter(dataLoader)
    for i in range(len_dataloader):
        img, label = data_iter.next()

        net.zero_grad()
        batch_size = len(label)

        if cuda:
            img = img.cuda()
            label = label.cuda()

        output = net(img)
        err = loss(output, label)
        err.backward()
        optimizer.step()
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], loss: %f'
                         % (epoch, i + 1, len_dataloader, err.data.cpu().item()))
        sys.stdout.flush()

    print('\n')
    acc = test(net)
    print('Accuracy of the dataset: %f' % acc)
    if acc > best_acc:
        best_acc = acc
        torch.save(net, os.path.join(logs_path, 'best_model.pth'))

print('============ Summary ============= \n')
print('Accuracy of the dataset: %f' % best_acc)

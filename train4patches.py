import os
import sys
import torch
import torch.nn as nn
from test4patches import test4patches
import torch.optim as optim
from model.LeNet4 import LeNet4
from dataUtils.getCifar10 import getCifar10
from torch.utils.data import DataLoader
from dataUtils.splitting import splittingTo4

# initialize para
logs_path = 'logs'
cuda = True if torch.cuda.is_available() else False
lr = 1e-3
batch_size = 128
image_size = 32
n_epoch = 20
weight_decay = 1e-5

# load model
net = LeNet4()

# load dataLoader
train_dataset = getCifar10(train=True)
patch1, patch2, patch3, patch4 = splittingTo4(train_dataset)

patch1Loader = DataLoader(dataset=patch1, batch_size=64, shuffle=True)
patch2Loader = DataLoader(dataset=patch2, batch_size=64, shuffle=True)
patch3Loader = DataLoader(dataset=patch3, batch_size=64, shuffle=True)
patch4Loader = DataLoader(dataset=patch4, batch_size=64, shuffle=True)

# initialize optimizer
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

loss = nn.CrossEntropyLoss()

if cuda:
    net = net.cuda()
    loss = loss.cuda()

best_acc = 0.0
for epoch in range(n_epoch):
    tot_loss = 0.0
    len_dataloader = len(patch1Loader)
    patch1_iter = iter(patch1Loader)
    patch2_iter = iter(patch2Loader)
    patch3_iter = iter(patch3Loader)
    patch4_iter = iter(patch4Loader)
    for i in range(len_dataloader):
        img_patch1, label = patch1_iter.next()
        img_patch2, _ = patch2_iter.next()
        img_patch3, _ = patch3_iter.next()
        img_patch4, _ = patch4_iter.next()

        net.zero_grad()
        batch_size = len(label)

        if cuda:
            img_patch1 = img_patch1.cuda()
            img_patch2 = img_patch2.cuda()
            img_patch3 = img_patch3.cuda()
            img_patch4 = img_patch4.cuda()
            label = label.cuda()

        output = net(img_patch1, img_patch2, img_patch3, img_patch4)
        err = loss(output, label)
        tot_loss += err
        err.backward()
        optimizer.step()
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], loss: %f'
                         % (epoch, i + 1, len_dataloader, err.data.cpu().item()))
        sys.stdout.flush()

    print('\n')
    tot_loss /= len_dataloader
    acc = test4patches(net)
    print('Accuracy of the dataset: %f  loss: %f' % (acc, tot_loss))
    if acc > best_acc:
        best_acc = acc
        torch.save(net, os.path.join(logs_path, '4patches_best_model.pth'))

print('============ Summary ============= \n')
print('Accuracy of the dataset: %f' % best_acc)
